import jax
import jax.numpy as jnp
import numpy as np

import qpax


def loss(Q, q, A, b, G, h):
    x = qpax.solve_qp_primal(Q, q, A, b, G, h)
    x_bar = jnp.ones(len(q))
    return jnp.dot(x - x_bar, x - x_bar)


# gradient of loss function
loss_grad = jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5))

# compatible with jit
loss_grad_jit = jax.jit(loss_grad)

n = 5  # size of x
m = 10  # rows in F

# create data for N_qps random nnls problems
Fs = jnp.array(np.random.randn(m, n))
gs = jnp.array(np.random.randn(m))


def form_qp(F, g):
    # convert the least squares to qp form
    n = F.shape[1]
    Q = F.T @ F
    q = -F.T @ g
    G = -jnp.eye(n)
    h = jnp.zeros(n)
    A = jnp.zeros((0, n))
    b = jnp.zeros(0)
    return Q, q, A, b, G, h


# create the QPs in a batched fashion
Q, q, A, b, G, h = form_qp(Fs, gs)

# calculate derivatives
derivs = loss_grad_jit(Q, q, A, b, G, h)
dl_dQ, dl_dq, dl_dA, dl_db, dl_dG, dl_dh = derivs
