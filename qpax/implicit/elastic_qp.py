"""Elastic QP solver using the implicit (retraction-manifold) PDIP backend."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from qpax._verbose import print_footer, print_header

from .pdip import (
    LinearSolver,
    SolverParams,
    _factor_init,
    _solve_init,
    derivative_retraction_map,
    derivative_retraction_map_kappa,
    ort_linesearch,
    retraction_map,
)


class ElasticQPData(NamedTuple):
    Q: jax.Array
    q: jax.Array
    G: jax.Array
    h: jax.Array
    penalty: jax.Array


class ElasticQPState(NamedTuple):
    x: jax.Array
    t: jax.Array
    s1: jax.Array
    s2: jax.Array
    z1: jax.Array
    z2: jax.Array


class FoldedElasticKKT(NamedTuple):
    """Factorization and diagonal terms for the folded elastic KKT solve.

    ``chol`` is the upper Cholesky factor of the primal Schur complement
    ``H = Q + Gᵀ diag(B1p * B2p / d) G``. ``H`` is symmetric positive definite
    whenever ``Q`` is (the ``Gᵀ diag(w) G`` term is PSD because ``w > 0``), so a
    Cholesky factorization is well defined and ~2x cheaper than the LU it
    replaces — consistent with the ``LinearSolver.CHOLESKY`` default used
    elsewhere in the backend.
    """

    B1n: jax.Array
    B2n: jax.Array
    denominator: jax.Array
    chol: jax.Array


# ------------------------------ initialization ------------------------------ #


def solve_init_elastic_ls(qp: ElasticQPData, solver: LinearSolver):
    """Least-squares warm start for the elastic primal and dual variables."""
    Q, q, G, h, penalty = qp
    ns = len(h)
    r1 = -q
    r2 = penalty * jnp.ones(ns, dtype=Q.dtype)
    r4 = h

    L_H = _factor_init(Q + 0.5 * G.T @ G, solver)
    x = _solve_init(L_H, r1 - 0.5 * G.T @ (r2 - r4), solver)
    z2 = 0.5 * (G @ x + r2 - r4)
    z1 = r2 - z2
    t = -z1

    x_big = jnp.concatenate((x, t))
    z_big = jnp.concatenate((z1, z2))

    return x_big, z_big


def initialize_elastic(
    qp: ElasticQPData,
    solver: LinearSolver = LinearSolver.CHOLESKY,
) -> ElasticQPState:
    """Shift the LS warm start so slack and dual variables are strictly positive."""
    x_big, z_big = solve_init_elastic_ls(qp, solver)

    alpha_p = -jnp.min(-z_big)
    s_big = jnp.where(alpha_p < 0, -z_big, -z_big + (1 + alpha_p))

    alpha_d = -jnp.min(z_big)
    z_big = jnp.where(alpha_d >= 0, z_big + (1 + alpha_d), z_big)

    nx = len(qp.q)
    ns = len(qp.h)
    return ElasticQPState(
        x=x_big[:nx],
        t=x_big[nx:],
        s1=s_big[:ns],
        s2=s_big[ns:],
        z1=z_big[:ns],
        z2=z_big[ns:],
    )


# ---------------------------- linear system solve --------------------------- #


def factorize_elastic_implicit_kkt(Q, G, v1, v2, kappa):
    """Factorize the primal Schur complement of the elastic KKT system.

    The ``(dt, dv1, dv2)`` equations are independent for every constraint.
    Eliminating those variables leaves the ``len(q)`` square system

        H = Q + G.T @ diag(B1p * B2p / d) @ G,

    where ``d = B1p * B2n + B2p * B1n``. This avoids factoring the
    ``len(q) + 3 * len(h)`` square block used by the unreduced formulation.
    """
    B1n_vec = derivative_retraction_map(-v1, kappa)
    B2n_vec = derivative_retraction_map(-v2, kappa)
    B1p_vec = derivative_retraction_map(v1, kappa)
    B2p_vec = derivative_retraction_map(v2, kappa)
    c1_vec = derivative_retraction_map_kappa(v1, kappa)
    c2_vec = derivative_retraction_map_kappa(v2, kappa)

    denominator = B1p_vec * B2n_vec + B2p_vec * B1n_vec
    schur_diagonal = B1p_vec * B2p_vec / denominator
    H = Q + jnp.matmul(
        G.T,
        schur_diagonal[:, None] * G,
        precision=jax.lax.Precision.HIGHEST,
    )
    # H is SPD but float32 cholesky can still fail (likely, due to rounding
    # errors leading to taking a sqrt of a negative pivot)
    # So, at the expense of one more factorization, use the unmodified
    # factorization if nan-free, otherwise use a regularized version
    chol, _ = jsp.linalg.cho_factor(H, lower=False)
    n = H.shape[0]
    delta = 10 * n * jnp.finfo(H.dtype).eps * jnp.max(jnp.diagonal(H))
    chol_shifted, _ = jsp.linalg.cho_factor(
        H + delta * jnp.eye(n, dtype=H.dtype), lower=False
    )
    chol = jnp.where(jnp.all(jnp.isfinite(chol)), chol, chol_shifted)
    factor = FoldedElasticKKT(B1n_vec, B2n_vec, denominator, chol)

    return B1p_vec, B2p_vec, c1_vec, c2_vec, factor


def solve_elastic_implicit_kkt_rhs(
    G,
    B1p_vec,
    B2p_vec,
    c1_vec,
    c2_vec,
    factor,
    rx,
    rt,
    rg1,
    rg2,
    rz1,
    rs1,
    rz2,
    rs2,
    rk,
):
    """Solve a folded implicit elastic KKT system and back-substitute."""
    B1n_vec, B2n_vec, denominator, chol = factor

    # Right-hand sides for the two primal constraints and t stationarity.
    R3 = -rg1 + rs1 + c1_vec * rk
    R4 = -rg2 + rs2 + c2_vec * rk
    St = rt + rz1 + rz2 + (c1_vec + c2_vec) * rk

    # The part of B2p * dv2 that is independent of G @ dx.
    dz2_offset = (B2p_vec / denominator) * (B1n_vec * St + B1p_vec * (R3 - R4))
    rhs = -rx + G.T @ (rz2 + c2_vec * rk - dz2_offset)
    dx = jsp.linalg.cho_solve((chol, False), rhs)

    Gdx = G @ dx
    dt = (
        B2p_vec * B1n_vec * Gdx
        - B1n_vec * B2n_vec * St
        - B1p_vec * B2n_vec * R3
        - B2p_vec * B1n_vec * R4
    ) / denominator
    common = Gdx + R3 - R4
    dv1 = (B2n_vec * St - B2p_vec * common) / denominator
    dv2 = (B1n_vec * St + B1p_vec * common) / denominator

    dk = -rk
    ds1 = -rg1 + dt
    ds2 = -rg2 - Gdx + dt
    dz1 = -rz1 + B1p_vec * dv1 - c1_vec * rk
    dz2 = -rz2 + B2p_vec * dv2 - c2_vec * rk

    return dx, dt, ds1, ds2, dz1, dz2, dv1, dv2, dk


# ---------------------------------------------------------------------------- #
#                                    solver                                    #
# ---------------------------------------------------------------------------- #


def solve_qp_elastic(
    Q: jax.Array,
    q: jax.Array,
    G: jax.Array,
    h: jax.Array,
    penalty: jax.Array,
    solver_tol: float = 1e-3,
    max_iter: int = 30,
    sigma: float = 0.125,
    verbose: bool = False,
):
    """Solve an elastic QP using the retraction-manifold PDIP method."""
    Q = jnp.atleast_2d(Q)
    G = jnp.atleast_2d(G)
    Q = 0.5 * (Q + Q.T)

    params = SolverParams(tol=solver_tol, max_iter=max_iter)
    qp = ElasticQPData(Q, q, G, h, penalty)
    state = initialize_elastic(qp)

    def _step(inputs):
        qp, st, converged, pdip_iter = inputs
        Q, q, G, h, penalty = qp
        x, t, s1, s2, z1, z2 = st

        m = len(h)
        v1 = z1 - s1
        v2 = z2 - s2

        # Floor kappa similar to the handling in the explicit backend
        # to prevent nans under Cholesky float32
        eps = jnp.finfo(Q.dtype).eps
        # Rough order-of-magnitude clamping for the floor
        kappa_floor = jnp.clip(0.05 * params.tol, 10 * eps, jnp.sqrt(eps))
        kappa = jnp.maximum((jnp.dot(s1, z1) + jnp.dot(s2, z2)) / (2 * m), kappa_floor)

        r1 = Q @ x + q + G.T @ z2
        r2 = -z1 - z2 + penalty * jnp.ones(m, dtype=Q.dtype)
        r3 = s1 * z1
        r4 = s2 * z2
        r5 = -t + s1
        r6 = G @ x - t + s2 - h

        kkt_res = jnp.concatenate((r1, r2, r3, r4, r5, r6))
        converged = jnp.where(jnp.linalg.norm(kkt_res, ord=jnp.inf) < params.tol, 1, 0)

        rz1 = z1 - retraction_map(v1, kappa)
        rs1 = s1 - retraction_map(-v1, kappa)
        rz2 = z2 - retraction_map(v2, kappa)
        rs2 = s2 - retraction_map(-v2, kappa)

        B1p, B2p, c1, c2, factor = factorize_elastic_implicit_kkt(Q, G, v1, v2, kappa)

        # Floor the target as well to prevent undershoot
        kappa_target = jnp.maximum(sigma * kappa, kappa_floor)
        rk = kappa - kappa_target

        dx, dt, ds1, ds2, dz1, dz2, dv1, dv2, dk = solve_elastic_implicit_kkt_rhs(
            G,
            B1p,
            B2p,
            c1,
            c2,
            factor,
            r1,
            r2,
            r5,
            r6,
            rz1,
            rs1,
            rz2,
            rs2,
            rk,
        )

        alpha = 0.99 * jnp.min(
            jnp.array(
                [
                    ort_linesearch(s1, ds1),
                    ort_linesearch(s2, ds2),
                    ort_linesearch(z1, dz1),
                    ort_linesearch(z2, dz2),
                ]
            )
        )

        if verbose:
            rt = jnp.concatenate((r1, r2))
            ri = jnp.concatenate((r5, r6))
            print(
                f"{pdip_iter:3d}   {kappa:9.2e}   "
                f"{jnp.linalg.norm(rt, ord=jnp.inf):9.2e}   "
                f"{jnp.linalg.norm(r3, ord=jnp.inf):9.2e}  "
                f"{jnp.linalg.norm(r4, ord=jnp.inf):9.2e}  "
                f"{jnp.linalg.norm(ri, ord=jnp.inf):9.2e}    "
                f"{alpha:6.4f}   {sigma:9.4f}"
            )

        # Under vmap, the while loop runs until the slowest lane converges.
        # Freezing converged lanes avoids post-convergence drift in f32.
        take = converged == 0
        x_new = jnp.where(take, x + alpha * dx, x)
        t_new = jnp.where(take, t + alpha * dt, t)
        v1_new = jnp.where(take, v1 + alpha * dv1, v1)
        v2_new = jnp.where(take, v2 + alpha * dv2, v2)
        kappa_new = jnp.where(take, kappa + alpha * dk, kappa)
        z1_new = jnp.where(take, retraction_map(v1_new, kappa_new), z1)
        s1_new = jnp.where(take, retraction_map(-v1_new, kappa_new), s1)
        z2_new = jnp.where(take, retraction_map(v2_new, kappa_new), z2)
        s2_new = jnp.where(take, retraction_map(-v2_new, kappa_new), s2)

        # Similar non-finite guard to explicit backend --
        # don't let possible blowups or nans enter the state
        step_finite = jnp.all(
            jnp.stack(
                [
                    jnp.all(jnp.isfinite(a))
                    for a in (x_new, t_new, s1_new, s2_new, z1_new, z2_new)
                ]
            )
        )
        x_new = jnp.where(step_finite, x_new, x)
        t_new = jnp.where(step_finite, t_new, t)
        s1_new = jnp.where(step_finite, s1_new, s1)
        s2_new = jnp.where(step_finite, s2_new, s2)
        z1_new = jnp.where(step_finite, z1_new, z1)
        z2_new = jnp.where(step_finite, z2_new, z2)

        new_state = ElasticQPState(x_new, t_new, s1_new, s2_new, z1_new, z2_new)
        return (qp, new_state, converged, pdip_iter + 1)

    def _cond(inputs):
        _, _, converged, pdip_iter = inputs
        return jnp.logical_and(pdip_iter < params.max_iter, converged == 0)

    init = (qp, state, 0, 0)
    if verbose:
        print_header(
            n=Q.shape[0],
            m=0,
            p=G.shape[0],
            tol=solver_tol,
            max_iter=max_iter,
            precision="f32" if Q.dtype == jnp.float32 else "f64",
            backend="implicit",
            sigma=sigma,
        )
        print(
            "iter     κ            rt          rc1        rc2         ri           α          σ"  # noqa: E501
        )
        print(
            "----------------------------------------------------------------------------------------"
        )
        outputs = init
        while _cond(outputs):
            outputs = _step(outputs)
    else:
        outputs = jax.lax.while_loop(_cond, _step, init)

    _, final_state, converged, pdip_iter = outputs
    x, t, s1, s2, z1, z2 = final_state

    if verbose:
        cost = 0.5 * x @ Q @ x + q @ x + penalty * jnp.sum(t)
        print_footer(converged, cost, pdip_iter)

    return x, t, s1, s2, z1, z2, converged, pdip_iter


# ---------------------------------------------------------------------------- #
#                                relaxed solver                                #
# ---------------------------------------------------------------------------- #


def pdip_newton_step_elastic(inputs, verbose: bool = False):
    """One relaxed Newton step for the elastic relaxed QP."""
    (
        Q,
        q,
        G,
        h,
        penalty,
        x,
        t,
        s1,
        s2,
        z1,
        z2,
        solver_tol,
        converged,
        pdip_iter,
        target_kappa,
        _B1p_prev,
        _B2p_prev,
        _c1_prev,
        _c2_prev,
        _factor_prev,
    ) = inputs

    m = len(h)
    v1 = z1 - s1
    v2 = z2 - s2

    # Similar flooring strategy as in solve_qp_elastic but without scaling with tol
    kappa_floor = jnp.maximum(jnp.sqrt(jnp.finfo(Q.dtype).eps), 1e-14)
    kappa = jnp.maximum((jnp.dot(s1, z1) + jnp.dot(s2, z2)) / (2 * m), kappa_floor)

    r1 = Q @ x + q + G.T @ z2
    r2 = -z1 - z2 + penalty * jnp.ones(m, dtype=Q.dtype)
    r3 = s1 * z1 - target_kappa
    r4 = s2 * z2 - target_kappa
    r5 = -t + s1
    r6 = G @ x - t + s2 - h

    kkt_res = jnp.concatenate((r1, r2, r3, r4, r5, r6))
    converged = jnp.where(jnp.linalg.norm(kkt_res, ord=jnp.inf) < solver_tol, 1, 0)

    rz1 = z1 - retraction_map(v1, kappa)
    rs1 = s1 - retraction_map(-v1, kappa)
    rz2 = z2 - retraction_map(v2, kappa)
    rs2 = s2 - retraction_map(-v2, kappa)

    B1p, B2p, c1, c2, factor = factorize_elastic_implicit_kkt(Q, G, v1, v2, kappa)

    rk = kappa - target_kappa
    dx, dt, ds1, ds2, dz1, dz2, dv1, dv2, dk = solve_elastic_implicit_kkt_rhs(
        G,
        B1p,
        B2p,
        c1,
        c2,
        factor,
        r1,
        r2,
        r5,
        r6,
        rz1,
        rs1,
        rz2,
        rs2,
        rk,
    )

    alpha = 0.99 * jnp.min(
        jnp.array(
            [
                ort_linesearch(s1, ds1),
                ort_linesearch(s2, ds2),
                ort_linesearch(z1, dz1),
                ort_linesearch(z2, dz2),
            ]
        )
    )

    if verbose:
        rt = jnp.concatenate((r1, r2))
        ri = jnp.concatenate((r5, r6))
        print(
            f"{pdip_iter:3d}   {kappa:9.2e}   "
            f"{jnp.linalg.norm(rt, ord=jnp.inf):9.2e}   "
            f"{jnp.linalg.norm(r3, ord=jnp.inf):9.2e}   "
            f"{jnp.linalg.norm(r4, ord=jnp.inf):9.2e}   "
            f"{jnp.linalg.norm(ri, ord=jnp.inf):9.2e}   "
            f"{alpha:6.4f}   {target_kappa:9.2e}"
        )

    # Under vmap, the while loop runs until the slowest lane converges.
    # Freezing converged lanes avoids post-convergence drift in f32.
    take = converged == 0
    x_new = jnp.where(take, x + alpha * dx, x)
    t_new = jnp.where(take, t + alpha * dt, t)
    v1_new = jnp.where(take, v1 + alpha * dv1, v1)
    v2_new = jnp.where(take, v2 + alpha * dv2, v2)
    kappa_new = jnp.where(take, kappa + alpha * dk, kappa)
    z1_new = jnp.where(take, retraction_map(v1_new, kappa_new), z1)
    s1_new = jnp.where(take, retraction_map(-v1_new, kappa_new), s1)
    z2_new = jnp.where(take, retraction_map(v2_new, kappa_new), z2)
    s2_new = jnp.where(take, retraction_map(-v2_new, kappa_new), s2)

    return (
        Q,
        q,
        G,
        h,
        penalty,
        x_new,
        t_new,
        s1_new,
        s2_new,
        z1_new,
        z2_new,
        solver_tol,
        converged,
        pdip_iter + 1,
        target_kappa,
        B1p,
        B2p,
        c1,
        c2,
        factor,
    )


def relax_qp_elastic(
    Q,
    q,
    G,
    h,
    penalty,
    x,
    t,
    s1,
    s2,
    z1,
    z2,
    solver_tol: float = 1e-5,
    target_kappa: float = 1e-3,
    max_iter: int = 30,
    sigma: float = 0.125,
    verbose: bool = False,
):
    """Relaxed elastic solve that also returns the last Newton-step factorization."""
    solver_tol = jnp.asarray(solver_tol, dtype=Q.dtype)
    target_kappa = jnp.asarray(target_kappa, dtype=Q.dtype)

    def relaxed_continuation_criteria(inputs):
        converged = inputs[12]
        pdip_iter = inputs[13]
        return jnp.logical_and(pdip_iter < max_iter, converged == 0)

    nz = G.shape[0]
    nx = G.shape[1]
    # Placeholder factor for the while_loop carry; overwritten on iteration 0.
    empty_factor = FoldedElasticKKT(
        jnp.zeros(nz, dtype=Q.dtype),
        jnp.zeros(nz, dtype=Q.dtype),
        jnp.ones(nz, dtype=Q.dtype),
        jnp.zeros((nx, nx), dtype=Q.dtype),
    )

    init_inputs = (
        Q,
        q,
        G,
        h,
        penalty,
        x,
        t,
        s1,
        s2,
        z1,
        z2,
        solver_tol,
        0,
        0,
        target_kappa,
        jnp.zeros(nz, dtype=Q.dtype),
        jnp.zeros(nz, dtype=Q.dtype),
        jnp.zeros(nz, dtype=Q.dtype),
        jnp.zeros(nz, dtype=Q.dtype),
        empty_factor,
    )

    if verbose:
        print_header(
            n=Q.shape[0],
            m=0,
            p=G.shape[0],
            tol=solver_tol,
            max_iter=max_iter,
            precision="f32" if Q.dtype == jnp.float32 else "f64",
            backend="implicit",
            sigma=sigma,
        )
        print(
            "iter      κ          rt          rc1        rc2         ri"
            "         alpha      target"
        )
        print(
            "-----------------------------------------------------------------------------------------------"
        )
        outputs = init_inputs
        while relaxed_continuation_criteria(outputs):
            outputs = pdip_newton_step_elastic(outputs, verbose=True)
    else:
        outputs = jax.lax.while_loop(
            relaxed_continuation_criteria, pdip_newton_step_elastic, init_inputs
        )

    x_rlx, t_rlx, s1_rlx, s2_rlx, z1_rlx, z2_rlx = outputs[5:11]
    converged = outputs[12]
    pdip_iter = outputs[13]
    B1p, B2p, c1, c2 = outputs[15:19]
    L_J = outputs[19]

    if verbose:
        cost = 0.5 * x_rlx @ Q @ x_rlx + q @ x_rlx + penalty * jnp.sum(t_rlx)
        print_footer(converged, cost, pdip_iter)

    return (
        x_rlx,
        t_rlx,
        s1_rlx,
        s2_rlx,
        z1_rlx,
        z2_rlx,
        B1p,
        B2p,
        c1,
        c2,
        L_J,
        converged,
        pdip_iter,
    )


# ---------------------------------------------------------------------------- #
#                              differentiation                                 #
# ---------------------------------------------------------------------------- #


def implicit_derivatives_elastic(dx, dz2, x, z2):
    dl_dQ = 0.5 * (jnp.outer(dx, x) + jnp.outer(x, dx))
    dl_dG = jnp.outer(dz2, x) + jnp.outer(z2, dx)

    dl_dq = dx
    dl_dh = -dz2

    return dl_dQ, dl_dq, dl_dG, dl_dh


def diff_qp_elastic(G, h, x, z2, kappa, B1p, B2p, c1, c2, L_J, dl_dx):
    zns = jnp.zeros_like(h)
    dx, _, _, _, _, dz2, _, _, _ = solve_elastic_implicit_kkt_rhs(
        G,
        B1p,
        B2p,
        c1,
        c2,
        L_J,
        dl_dx,
        zns,
        zns,
        zns,
        zns,
        zns,
        zns,
        zns,
        jnp.zeros_like(kappa),
    )

    return implicit_derivatives_elastic(dx, dz2, x, z2)


@jax.custom_vjp
def solve_qp_elastic_primal(
    Q, q, G, h, penalty, solver_tol=1e-5, target_kappa=1e-3, max_iter=30
):
    x, _, _, _, _, _, _, _ = solve_qp_elastic(
        Q, q, G, h, penalty, solver_tol=solver_tol, max_iter=max_iter
    )
    return x


"""
these two functions are only called when we diff solve_qp_elastic_primal
"""


def solve_qp_elastic_primal_forward(
    Q, q, G, h, penalty, solver_tol=1e-5, target_kappa=1e-3, max_iter=30
):
    x, t, s1, s2, z1, z2, _, _ = solve_qp_elastic(
        Q, q, G, h, penalty, solver_tol=solver_tol, max_iter=max_iter
    )

    (
        xr,
        _tr,
        _s1r,
        _s2r,
        _z1r,
        z2r,
        B1p,
        B2p,
        c1,
        c2,
        L_J,
        _,
        _,
    ) = relax_qp_elastic(
        Q,
        q,
        G,
        h,
        penalty,
        x,
        t,
        s1,
        s2,
        z1,
        z2,
        solver_tol=solver_tol,
        target_kappa=target_kappa,
        max_iter=max_iter,
    )

    res = (G, h, xr, z2r, target_kappa, B1p, B2p, c1, c2, L_J)
    return x, res


def solve_qp_elastic_primal_backward(res, input_grad):
    G, h, xr, z2r, kappa, B1p, B2p, c1, c2, L_J = res

    dl_dQ, dl_dq, dl_dG, dl_dh = diff_qp_elastic(
        G, h, xr, z2r, kappa, B1p, B2p, c1, c2, L_J, input_grad
    )

    return (dl_dQ, dl_dq, dl_dG, dl_dh, None, None, None, None)


solve_qp_elastic_primal.defvjp(
    solve_qp_elastic_primal_forward, solve_qp_elastic_primal_backward
)
