import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import grad

from qpax import solve_qp_primal

from .misc_test_utils import finite_difference, generate_random_qp

jax.config.update("jax_enable_x64", True)


def _make_my_f(backend):
    @functools.partial(jax.jit, static_argnames=())
    def my_f(Q, q, A, b, G, h):
        x = solve_qp_primal(Q, q, A, b, G, h, backend=backend, target_kappa=1e-3)
        x_bar = jnp.ones(len(q))
        return jnp.dot(x - x_bar, x - x_bar)

    return my_f


@pytest.mark.parametrize("backend", ["e", "i"])
def test_derivs(backend):
    np.random.seed(3)
    nx = 15
    ns = 10
    ny = 3
    Q, q, A, b, G, h, x_true, s_true, z_true, y_true = generate_random_qp(nx, ns, ny)

    del x_true, s_true, z_true, y_true

    my_f = _make_my_f(backend)

    def my_f_select(inputs, X, i):
        new_inputs = tuple(
            X if index == i else value for index, value in enumerate(inputs)
        )
        return my_f(*new_inputs)

    inputs = (Q, q, A, b, G, h)
    grad_my_f = jax.jit(grad(my_f, argnums=(0, 1, 2, 3, 4, 5)))
    derivs = grad_my_f(*inputs)

    input_names = ("Q", "q", "A", "b", "G", "h")
    for i in range(6):
        print("-------------input: ", input_names[i], "----------------")

        def lambda_f(_X, _i=i):
            return my_f_select(inputs, _X, _i)

        fd_deriv = finite_difference(lambda_f, inputs[i])

        assert fd_deriv.shape == derivs[i].shape

        print("fd_deriv_norm: ")
        print(jnp.linalg.norm(fd_deriv))
        print("error_norm: ", jnp.linalg.norm(derivs[i] - fd_deriv))

        assert jnp.linalg.norm(derivs[i] - fd_deriv) < (0.2 * jnp.linalg.norm(fd_deriv))


@pytest.mark.parametrize("backend", ["e", "i"])
def test_inequality_parameter_derivatives(backend):
    rng = np.random.default_rng(0)
    nx, ns = 4, 6

    factor = rng.standard_normal((nx, nx))
    Q = jnp.array(factor @ factor.T + np.eye(nx))
    q = jnp.array(rng.standard_normal(nx))
    G = rng.standard_normal((ns, nx))
    x0 = rng.standard_normal(nx)
    h = jnp.array(
        G @ x0 + np.array([0.0, 0.5, -0.1, 1.0, 0.3, -0.2])
    )
    G = jnp.array(G)
    A = jnp.zeros((0, nx))
    b = jnp.zeros(0)

    def loss(G_, h_):
        x = solve_qp_primal(
            Q,
            q,
            A,
            b,
            G_,
            h_,
            backend=backend,
            # note: tolerances assume x64
            solver_tol=1e-10,
            target_kappa=1e-8,
        )
        return 0.5 * jnp.sum(x**2)

    grad_G_ad, grad_h_ad = jax.grad(loss, argnums=(0, 1))(G, h)

    eps = 1e-6
    directions = jnp.eye(G.size).reshape((-1, *G.shape))
    grad_G_fd = jax.vmap(
        lambda direction: (
            loss(G + eps * direction, h) - loss(G - eps * direction, h)
        )
        / (2 * eps)
    )(directions).reshape(G.shape)

    directions = jnp.eye(h.size)
    grad_h_fd = jax.vmap(
        lambda direction: (
            loss(G, h + eps * direction) - loss(G, h - eps * direction)
        )
        / (2 * eps)
    )(directions)

    np.testing.assert_allclose(grad_G_ad, grad_G_fd, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grad_h_ad, grad_h_fd, rtol=1e-5, atol=1e-6)
