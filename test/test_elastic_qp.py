"""Tests for the elastic QP backends."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import qpax
from qpax.implicit.elastic_qp import (
    factorize_elastic_implicit_kkt,
    relax_qp_elastic,
    solve_elastic_implicit_kkt_rhs,
)


@pytest.mark.parametrize("backend", ["e", "i"])
def test_elastic_smoke(backend):
    Q = jnp.eye(2)
    q = jnp.zeros(2)
    G = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    h = jnp.array([1.0, 1.0])
    penalty = jnp.array(1.0)

    out = qpax.solve_qp_elastic(Q, q, G, h, penalty, backend=backend)
    x = out[0]
    assert x.shape == (2,)
    assert jnp.all(jnp.isfinite(x))
    assert int(out[-2]) == 1


@pytest.mark.parametrize("backend", ["e", "i"])
def test_elastic_preserves_state_on_terminal_iteration(backend):
    Q = jnp.eye(2, dtype=jnp.float32)
    q = jnp.zeros(2, dtype=jnp.float32)
    G = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    h = jnp.array([1.0, 1.0], dtype=jnp.float32)
    penalty = jnp.array(1.0, dtype=jnp.float32)

    full = qpax.solve_qp_elastic(Q, q, G, h, penalty, backend=backend)
    iters = int(full[-1])

    assert int(full[-2]) == 1
    assert iters > 1

    capped = qpax.solve_qp_elastic(
        Q, q, G, h, penalty, backend=backend, max_iter=iters - 1
    )

    for full_value, capped_value in zip(full[:6], capped[:6], strict=True):
        np.testing.assert_array_equal(np.asarray(full_value), np.asarray(capped_value))


def _dense_elastic_reference(Q, G, B1p, B2p, c1, c2, factor, residuals):
    """Solve the unreduced (n + 3p) elastic KKT block directly.

    Mirrors the block system the folded Schur solve replaces, so the folded
    directions can be checked against it.
    """
    n, p = G.shape[1], G.shape[0]
    rx, rt, rg1, rg2, rz1, rs1, rz2, rs2, rk = residuals
    a1 = rg1 + rz1 - rs1
    a2 = rg2 + rz2 - rs2
    zeros_pp = jnp.zeros((p, p), dtype=Q.dtype)
    zeros_np = jnp.zeros((n, p), dtype=Q.dtype)
    zeros_pn = jnp.zeros((p, n), dtype=Q.dtype)
    eye_p = jnp.eye(p, dtype=Q.dtype)
    dense_kkt = jnp.block(
        [
            [Q - G.T @ G, G.T, zeros_np, G.T],
            [G, -2.0 * eye_p, -eye_p, -eye_p],
            [zeros_pn, -eye_p, -jnp.diag(factor.B1n), zeros_pp],
            [G, -eye_p, zeros_pp, -jnp.diag(factor.B2n)],
        ]
    )
    dense_rhs = jnp.concatenate(
        [
            rx - G.T @ a2,
            rt + a1 + a2,
            rg1 - rs1 - c1 * rk,
            rg2 - rs2 - c2 * rk,
        ]
    )
    dense_solution = jnp.linalg.solve(dense_kkt, -dense_rhs)
    dx = dense_solution[:n]
    dt = dense_solution[n : n + p]
    dv1 = dense_solution[n + p : n + 2 * p]
    dv2 = dense_solution[n + 2 * p :]
    dense = (
        dx,
        dt,
        -rg1 + dt,
        -rg2 - G @ dx + dt,
        -rz1 + B1p * dv1 - c1 * rk,
        -rz2 + B2p * dv2 - c2 * rk,
        dv1,
        dv2,
        -rk,
    )
    return dense, dense_kkt


def test_implicit_elastic_folded_kkt_matches_dense_system():
    """The n-by-n Schur solve must match the former (n + 3p)-block solve."""
    n, p = 5, 17
    keys = iter(jax.random.split(jax.random.PRNGKey(11), 13))

    R = jax.random.normal(next(keys), (n, n))
    Q = R.T @ R + jnp.eye(n)
    G = jax.random.normal(next(keys), (p, n))
    v1 = jax.random.normal(next(keys), (p,))
    v2 = jax.random.normal(next(keys), (p,))
    kappa = jnp.float32(0.2)
    residuals = (
        jax.random.normal(next(keys), (n,)),
        *(jax.random.normal(next(keys), (p,)) for _ in range(7)),
        jnp.float32(0.07),
    )

    B1p, B2p, c1, c2, factor = factorize_elastic_implicit_kkt(Q, G, v1, v2, kappa)
    folded = solve_elastic_implicit_kkt_rhs(G, B1p, B2p, c1, c2, factor, *residuals)
    dense, dense_kkt = _dense_elastic_reference(
        Q, G, B1p, B2p, c1, c2, factor, residuals
    )

    assert factor.chol.shape == (n, n)
    assert dense_kkt.shape == (n + 3 * p, n + 3 * p)
    for folded_value, dense_value in zip(folded, dense, strict=True):
        np.testing.assert_allclose(folded_value, dense_value, rtol=1e-3, atol=1e-3)


def test_implicit_elastic_folded_cholesky_handles_psd_q():
    """Cholesky must stay finite when Q is PSD (rank-deficient), not just PD.

    ``H = Q + Gᵀ diag(w) G`` is still SPD here because the full-column-rank ``G``
    makes the second term positive definite over ``Q``'s nullspace, but the
    ``+I`` cushion of the other tests is removed. This is the case that would
    expose a fragile factorization: ``cho_factor`` returns ``nan`` rather than
    raising on a non-PD input, so a silent loss of definiteness would surface
    here as non-finite directions.
    """
    n, p, rank = 6, 30, 3
    keys = iter(jax.random.split(jax.random.PRNGKey(5), 13))

    # Rank-deficient PSD Q: R is (n, rank), so Q has n - rank zero eigenvalues.
    R = jax.random.normal(next(keys), (n, rank), dtype=jnp.float32)
    Q = R @ R.T
    smallest_eig = float(jnp.linalg.eigvalsh(Q).min())
    assert smallest_eig < 1e-4  # Q really is (numerically) singular, not PD

    G = jax.random.normal(next(keys), (p, n), dtype=jnp.float32)  # full column rank
    v1 = jax.random.normal(next(keys), (p,), dtype=jnp.float32)
    v2 = jax.random.normal(next(keys), (p,), dtype=jnp.float32)
    kappa = jnp.float32(0.2)
    residuals = (
        jax.random.normal(next(keys), (n,), dtype=jnp.float32),
        *(jax.random.normal(next(keys), (p,), dtype=jnp.float32) for _ in range(7)),
        jnp.float32(0.07),
    )

    B1p, B2p, c1, c2, factor = factorize_elastic_implicit_kkt(Q, G, v1, v2, kappa)
    folded = solve_elastic_implicit_kkt_rhs(G, B1p, B2p, c1, c2, factor, *residuals)

    # Cholesky did not silently produce nan/inf on the PSD (non-PD) system.
    assert bool(jnp.all(jnp.isfinite(factor.chol)))
    for value in folded:
        assert bool(jnp.all(jnp.isfinite(value)))

    # And the folded directions still match the dense reference.
    dense, _ = _dense_elastic_reference(Q, G, B1p, B2p, c1, c2, factor, residuals)
    for folded_value, dense_value in zip(folded, dense, strict=True):
        np.testing.assert_allclose(folded_value, dense_value, rtol=1e-3, atol=1e-3)


def test_implicit_elastic_psd_q_solves_end_to_end():
    """A full f32 solve + backward pass must survive a rank-deficient PSD Q.

    Exercises the while-loop carry and implicit-diff path (not just the isolated
    factorization) when the Cholesky factor is built from a singular Q. The
    primal solution is not asserted against a target: a rank-deficient Q leaves
    a flat direction, so only the KKT residual is a meaningful accuracy check.
    """
    n, p, rank = 8, 40, 3
    keys = jax.random.split(jax.random.PRNGKey(3), 4)
    R = jax.random.normal(keys[0], (n, rank), dtype=jnp.float32)
    Q = R @ R.T  # PSD, rank 3 -> 5 zero eigenvalues
    assert float(jnp.linalg.eigvalsh(Q).min()) < 1e-4

    G = jax.random.normal(keys[1], (p, n), dtype=jnp.float32)  # full column rank
    x_star = 0.2 * jax.random.normal(keys[2], (n,), dtype=jnp.float32)
    magnitude = 0.25 + 0.5 * jax.random.uniform(keys[3], (p,), dtype=jnp.float32)
    violated = jnp.arange(p) % 2 == 0
    t = jnp.where(violated, magnitude, 0.0)
    s2 = jnp.where(violated, 0.0, magnitude)
    penalty = jnp.float32(1.0)
    z2 = jnp.where(violated, penalty, 0.0)
    h = G @ x_star - t + s2
    q = -Q @ x_star - G.T @ z2

    x, tt, s1, s2o, z1, z2o, converged, _ = qpax.solve_qp_elastic(
        Q, q, G, h, penalty, backend="i", solver_tol=1e-4, max_iter=60
    )
    residual = jnp.concatenate(
        (
            Q @ x + q + G.T @ z2o,
            -z1 - z2o + penalty,
            s1 * z1,
            s2o * z2o,
            -tt + s1,
            G @ x - tt + s2o - h,
        )
    )
    assert int(converged) == 1
    assert bool(jnp.all(jnp.isfinite(x)))
    assert float(jnp.linalg.norm(residual, ord=jnp.inf)) < 1e-3

    def loss(q_value):
        x_primal = qpax.solve_qp_elastic_primal(
            Q,
            q_value,
            G,
            h,
            penalty,
            backend="i",
            solver_tol=1e-4,
            target_kappa=1e-3,
            max_iter=60,
        )
        return jnp.sum(x_primal * x_primal)

    gradient = jax.jit(jax.grad(loss))(q)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def _known_solution_elastic_qp(n, p):
    keys = jax.random.split(jax.random.PRNGKey(1000 + p), 4)
    R = jax.random.normal(keys[0], (n, n), dtype=jnp.float32)
    Q = R.T @ R / n + jnp.eye(n, dtype=jnp.float32)
    x = 0.2 * jax.random.normal(keys[1], (n,), dtype=jnp.float32)
    G = jax.random.normal(keys[2], (p, n), dtype=jnp.float32) / jnp.sqrt(n)
    magnitude = 0.25 + 0.5 * jax.random.uniform(keys[3], (p,), dtype=jnp.float32)
    violated = jnp.arange(p) % 2 == 0
    t = jnp.where(violated, magnitude, 0.0)
    s2 = jnp.where(violated, 0.0, magnitude)
    penalty = jnp.float32(1.0)
    z2 = jnp.where(violated, penalty, 0.0)
    h = G @ x - t + s2
    q = -Q @ x - G.T @ z2
    return Q, q, G, h, penalty, x


@pytest.mark.parametrize("n_constraints", [1, 50, 500])
def test_implicit_elastic_f32_constraint_scaling_accuracy(n_constraints):
    Q, q, G, h, penalty, expected_x = _known_solution_elastic_qp(35, n_constraints)
    x, t, s1, s2, z1, z2, converged, _ = qpax.solve_qp_elastic(
        Q,
        q,
        G,
        h,
        penalty,
        backend="i",
        solver_tol=1e-3,
        max_iter=50,
    )

    residual = jnp.concatenate(
        (
            Q @ x + q + G.T @ z2,
            -z1 - z2 + penalty,
            s1 * z1,
            s2 * z2,
            -t + s1,
            G @ x - t + s2 - h,
        )
    )
    assert int(converged) == 1
    assert float(jnp.linalg.norm(residual, ord=jnp.inf)) < 1e-3
    np.testing.assert_allclose(x, expected_x, rtol=5e-3, atol=3e-3)


def _ill_conditioned_elastic_batch(batch, n, p, seed):
    """Elastic QPs whose folded Schur complement stresses f32 Cholesky.

    A large penalty puts ~penalty^2 / kappa on the Schur diagonal at active
    constraints, driving cond(H) toward the f32 Cholesky limit ~1/(n*eps).
    A quarter of the rows are infeasible at the seed point so the elastic
    slack activates and both dual pairs sit near the penalty.
    """
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(batch, n, n))
    Q = np.einsum("bij,bik->bjk", M, M) / n + 1e-1 * np.eye(n)[None]
    q = rng.normal(size=(batch, n))
    G = rng.normal(size=(batch, p, n)) * 0.5
    x0 = rng.normal(size=(batch, n)) * 0.5
    slack = rng.uniform(0.1, 1.0, size=(batch, p))
    slack[:, : max(1, p // 4)] *= -1.0
    h = np.einsum("bij,bj->bi", G, x0) + slack
    penalty = jnp.float32(100.0)
    to_f32 = lambda a: jnp.asarray(a, dtype=jnp.float32)  # noqa: E731
    return to_f32(Q), to_f32(q), to_f32(G), to_f32(h), penalty


def test_implicit_elastic_f32_ill_conditioned_batch_stays_finite():
    """Batched f32 solve + relax must not emit NaN near the Cholesky limit.

    Regression test for two f32 failure modes in the implicit elastic backend:

    1. kappa collapsing below f32 epsilon under vmap
    2. cho_factor producing NaN on an SPD-but-ill-conditioned H
    """
    Q, q, G, h, penalty = _ill_conditioned_elastic_batch(128, 29, 96, seed=7)

    def pipeline(Qi, qi, Gi, hi):
        out = qpax.solve_qp_elastic(
            Qi, qi, Gi, hi, penalty, backend="i", solver_tol=1e-3, max_iter=50
        )
        x, t, s1, s2, z1, z2 = out[:6]
        relaxed = relax_qp_elastic(
            Qi,
            qi,
            Gi,
            hi,
            penalty,
            x,
            t,
            s1,
            s2,
            z1,
            z2,
            solver_tol=1e-3,
            target_kappa=1e-3,
            max_iter=50,
        )
        return out[0], out[6], relaxed[0], relaxed[11]

    x_fwd, conv_fwd, x_rlx, conv_rlx = jax.jit(jax.vmap(pipeline))(Q, q, G, h)

    assert bool(jnp.all(jnp.isfinite(x_fwd)))
    assert bool(jnp.all(jnp.isfinite(x_rlx)))
    assert float(jnp.mean(conv_fwd)) >= 0.98
    assert float(jnp.mean(conv_rlx)) >= 0.99


def test_implicit_elastic_folded_factor_supports_backward_pass():
    Q, q, G, h, penalty, _ = _known_solution_elastic_qp(5, 10)

    def loss(q_value):
        x = qpax.solve_qp_elastic_primal(
            Q,
            q_value,
            G,
            h,
            penalty,
            backend="i",
            solver_tol=1e-3,
            target_kappa=1e-3,
            max_iter=50,
        )
        return jnp.sum(x * x)

    gradient = jax.jit(jax.grad(loss))(q)
    assert gradient.shape == q.shape
    assert gradient.dtype == q.dtype
    assert bool(jnp.all(jnp.isfinite(gradient)))
