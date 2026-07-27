# Elastic

The elastic QP relaxes the inequality constraints of a convex QP with a
non-negative slack $t$ that is penalized in the cost,

$$
\underset{x,\, t}{\text{minimize}} \quad
\tfrac{1}{2}\, x^{\top} Q x + q^{\top} x + \rho\, \mathbf{1}^{\top} t
\quad \text{s.t.} \quad
G x \leq h + t, \;\; t \geq 0.
$$

Introducing slacks $s_1, s_2 \geq 0$ and multipliers $z_1, z_2 \geq 0$ for the
two inequalities $t \geq 0$ and $G x - t \leq h$, the KKT conditions are

$$
\begin{aligned}
Q x + q + G^{\top} z_2 &= 0, \\
\rho\, \mathbf{1} - z_1 - z_2 &= 0, \\
t - s_1 &= 0, \\
G x - t + s_2 - h &= 0, \\
(s_1)_i (z_1)_i &= 0, \qquad s_1, z_1 \geq 0, \\
(s_2)_i (z_2)_i &= 0, \qquad s_2, z_2 \geq 0.
\end{aligned}
$$

As in primal-dual interior-point methods, we replace the hard complementarities
$(s_k)_i (z_k)_i = 0$ with smooth conditions $(s_k)_i (z_k)_i = \mu$ and drive
$\mu \to 0$. The specific details of how the forward and backward passes are
conducted depend on the underlying explicit or implicit backend (`backend="e"`
or `backend="i"`).

## Forward pass

The forward pass first drives the iterate to a KKT point (*Solve QP*) and
then refines it to a $\kappa$-relaxed KKT point (*Relax QP*) so that the
implicit-function gradients used in the backward pass are well conditioned.

### Solve QP

=== "Explicit"

    ```text
    function solve_qp_elastic_explicit(Q, q, G, h, ρ; tol, max_iter)

        (x, t, s1, s2, z1, z2) ← initialize(Q, q, G, h, ρ)

        for i = 1, ..., max_iter do

            # Residuals and convergence check
            r ← evaluate_residuals(Q, q, G, h, ρ, x, t, s1, s2, z1, z2)
            if ‖r‖∞ < tol then
                return x, t, s1, s2, z1, z2
            end if

            # Factor the reduced KKT system once per iteration
            K̃ ← precompute_kkt_factors(Q, G, s1, z1, s2, z2)

            # 1. Predictor: affine step with μ = 0
            (Δxa, Δta, Δs1a, Δs2a, Δz1a, Δz2a) ← solve_kkt(K̃, r, μ = 0)
            αa ← linesearch(s1, s2, z1, z2, Δs1a, Δs2a, Δz1a, Δz2a)

            # 2. Centering parameter
            μ      ← (s1ᵀ z1 + s2ᵀ z2) / m
            μ_aff  ← ((s1 + αa Δs1a)ᵀ (z1 + αa Δz1a)
                      + (s2 + αa Δs2a)ᵀ (z2 + αa Δz2a)) / m
            σ      ← (μ_aff / μ)^3

            # 3. Corrector with second-order correction in complementarity
            (Δx, Δt, Δs1, Δs2, Δz1, Δz2)
                ← solve_kkt(K̃, r, σμ − (Δs1a ⊙ Δz1a, Δs2a ⊙ Δz2a))

            # 4. Step
            α ← linesearch(s1, s2, z1, z2, Δs1, Δs2, Δz1, Δz2)
            (x, t, s1, s2, z1, z2)
                ← (x + αΔx, t + αΔt,
                   s1 + αΔs1, s2 + αΔs2,
                   z1 + αΔz1, z2 + αΔz2)

        end for

    end function
    ```

=== "Implicit"

    ```text
    function solve_qp_elastic_implicit(Q, q, G, h, ρ; σ, tol, max_iter)

        (x, t, s1, s2, z1, z2) ← initialize(Q, q, G, h, ρ)

        for i = 1, ..., max_iter do

            # Manifold coordinates
            v1 ← z1 - s1
            v2 ← z2 - s2
            κ  ← (s1ᵀ z1 + s2ᵀ z2) / m

            # Residuals and convergence check
            r ← evaluate_residuals(Q, q, G, h, ρ, x, t, s1, s2, z1, z2)
            if ‖r‖∞ < tol then
                return x, t, s1, s2, z1, z2
            end if

            # Factor the reduced KKT system once per iteration
            K̃ ← precompute_kkt_factors(Q, G, v1, v2, κ)

            # Solve KKT
            κ_target ← σκ
            (Δx, Δt, Δs1, Δs2, Δz1, Δz2, Δv1, Δv2, Δκ)
                ← solve_kkt(K̃, r, κ_target)

            # Step
            α ← linesearch(s1, s2, z1, z2, Δs1, Δs2, Δz1, Δz2)

            x  ← x  + αΔx
            t  ← t  + αΔt
            v1 ← v1 + αΔv1
            v2 ← v2 + αΔv2
            κ  ← κ  + αΔκ

            # Retract back to positive slack-dual variables
            (z1, s1) ← retraction_map(v1, κ)
            (z2, s2) ← retraction_map(v2, κ)

        end for

    end function
    ```

### Solve KKT

Inside each Newton iteration, `solve_kkt` reduces the full elastic KKT
system to a smaller block in the primal-only unknowns, solves it, and
recovers the slack and dual directions by back-substitution.

=== "Explicit"

    ```text
    function solve_kkt_elastic_explicit(Q, G, s1, z1, s2, z2;
                                        rx, rt, rc1, rc2, rg1, rg2)

        # Reduce to a system in Δx alone
        a1 ← s1 ⊘ z1
        a2 ← s2 ⊘ z2
        a3 ← a1 + a2
        p  ← rg1 - rg2 + rc2 ⊘ z2 - rc1 ⊘ z1 - a1 ⊙ rt

        H   ← Q + Gᵀ diag(1 ⊘ a3) G
        L_H ← factor(H)
        Δx  ← solve(L_H, rx - Gᵀ (p ⊘ a3))

        # Back-substitution
        Δz2 ← (p + G Δx) ⊘ a3
        Δz1 ← -rt - Δz2
        Δs1 ← (rc1 - s1 ⊙ Δz1) ⊘ z1
        Δs2 ← (rc2 - s2 ⊙ Δz2) ⊘ z2
        Δt  ← Δs1 - rg1

        return Δx, Δt, Δs1, Δs2, Δz1, Δz2, L_H

    end function
    ```

=== "Implicit"

    ```text
    function solve_kkt_elastic_implicit(Q, G, v1, v2, κ;
                                        rx, rt, rg1, rg2,
                                        rz1, rs1, rz2, rs2, rκ)

        # Diagonal blocks from the retraction map
        B1⁻ ← diag(∂_v b_κ(-v1))
        B2⁻ ← diag(∂_v b_κ(-v2))
        B1⁺ ← diag(∂_v b_κ( v1))
        B2⁺ ← diag(∂_v b_κ( v2))
        c1  ← ∂_κ b_κ(v1)
        c2  ← ∂_κ b_κ(v2)

        # Each constraint has an independent 3-by-3 block in
        # (Δt_i, Δv1_i, Δv2_i), so eliminate those blocks.
        u1 ← -rg1 + rs1 + c1 ⊙ rκ
        u2 ← -rg2 + rs2 + c2 ⊙ rκ
        ut ← rt + rz1 + rz2 + (c1 + c2) ⊙ rκ
        d  ← B1⁺ B2⁻ + B2⁺ B1⁻
        w  ← diag(B1⁺ B2⁺) ⊘ diag(d)
        o2 ← B2⁺ d⁻¹ (B1⁻ ut + B1⁺ (u1 - u2))

        # Factor and solve an n-by-n primal system.
        H   ← Q + Gᵀ diag(w) G
        L_H ← factor(H)
        Δx  ← solve(L_H, -rx + Gᵀ (rz2 + c2 ⊙ rκ - o2))

        # Elementwise back-substitution
        y   ← G Δx
        Δt  ← (B2⁺ B1⁻ y - B1⁻ B2⁻ ut
               - B1⁺ B2⁻ u1 - B2⁺ B1⁻ u2) ⊘ d
        Δv1 ← (B2⁻ ut - B2⁺ (y + u1 - u2)) ⊘ d
        Δv2 ← (B1⁻ ut + B1⁺ (y + u1 - u2)) ⊘ d

        # Back-substitution
        Δκ  ← -rκ
        Δs1 ← -rg1 + Δt
        Δs2 ← -rg2 - G Δx + Δt
        Δz1 ← -rz1 + B1⁺ Δv1 - c1 rκ
        Δz2 ← -rz2 + B2⁺ Δv2 - c2 rκ

        return Δx, Δt, Δs1, Δs2, Δz1, Δz2, Δv1, Δv2, Δκ

    end function
    ```

### Relax QP

Once Solve QP has converged, the iterate sits at a KKT point where the
complementarity gap collapses to zero, causing the implicit-function gradients
used in differentiation to become non-informative. *Relax QP* drives the
iterate to a $\kappa$-relaxed KKT point instead, trading a controlled bias for
usable sensitivities.

=== "Explicit"

    ```text
    function relax_qp_elastic_explicit(Q, q, G, h, ρ, x, t, s1, s2, z1, z2;
                                       κ_relax, tol, max_iter)

        for i = 1, ..., max_iter do
            r ← evaluate_residuals(Q, q, G, h, ρ, x, t, s1, s2, z1, z2)
            if ‖r‖∞ < tol then
                return x, t, s1, s2, z1, z2
            end if

            K̃ ← precompute_kkt_factors(Q, G, s1, z1, s2, z2)
            (Δx, Δt, Δs1, Δs2, Δz1, Δz2) ← solve_kkt(K̃, r, κ_relax)
            α ← linesearch(s1, s2, z1, z2, Δs1, Δs2, Δz1, Δz2)

            (x, t, s1, s2, z1, z2)
                ← (x + αΔx, t + αΔt,
                   s1 + αΔs1, s2 + αΔs2,
                   z1 + αΔz1, z2 + αΔz2)
        end for

    end function
    ```

=== "Implicit"

    ```text
    function relax_qp_elastic_implicit(Q, q, G, h, ρ, x, t, s1, s2, z1, z2;
                                       κ_relax, tol, max_iter)

        for i = 1, ..., max_iter do
            v1 ← z1 - s1
            v2 ← z2 - s2
            κ  ← (s1ᵀ z1 + s2ᵀ z2) / m

            r ← evaluate_residuals(Q, q, G, h, ρ, x, t, s1, s2, z1, z2)
            if ‖r‖∞ < tol then
                return x, t, s1, s2, z1, z2
            end if

            if K̃ is not cached then
                K̃ ← precompute_kkt_factors(Q, G, v1, v2, κ)
            end if

            (Δx, Δt, Δs1, Δs2, Δz1, Δz2, Δv1, Δv2, Δκ)
                ← solve_kkt(K̃, r, κ_relax)
            α ← linesearch(s1, s2, z1, z2, Δs1, Δs2, Δz1, Δz2)

            x  ← x  + αΔx
            t  ← t  + αΔt
            v1 ← v1 + αΔv1
            v2 ← v2 + αΔv2
            κ  ← κ  + αΔκ

            (z1, s1) ← retraction_map(v1, κ)
            (z2, s2) ← retraction_map(v2, κ)
        end for

    end function
    ```

## Backward pass

The backward pass applies the implicit-function theorem at the relaxed KKT
point to compute gradients.

### Computing gradients

=== "Explicit"

    ```text
    function compute_qp_gradients_elastic_explicit(Q, q, G, h, ρ,
                                                   x, t, s1, s2, z1, z2,
                                                   κ_relax, ∇xl; K̃)

        r ← (-∇xl, 0, 0, 0, 0, 0)
        (dx, dt, ds1, ds2, dẑ1, dẑ2) ← solve_kkt(K̃, r, κ_relax)

        ∇Ql ← (dx xᵀ + x dxᵀ) / 2
        ∇ql ← dx
        ∇Gl ← dẑ2 xᵀ + z2 dxᵀ
        ∇hl ← -dẑ2

        return ∇Ql, ∇ql, ∇Gl, ∇hl

    end function
    ```

=== "Implicit"

    ```text
    function compute_qp_gradients_elastic_implicit(Q, q, G, h, ρ,
                                                   x, t, s1, s2, z1, z2,
                                                   κ_relax, ∇xl; K̃)

        r ← (-∇xl, 0, 0, 0, 0, 0, 0, 0, 0)
        (dx, dt, ds1, ds2, dz1, dz2, _, _, _)
            ← solve_kkt(K̃, r, κ_relax)

        ∇Ql ← (dx xᵀ + x dxᵀ) / 2
        ∇ql ← dx
        ∇Gl ← dz2 xᵀ + z2 dxᵀ
        ∇hl ← -dz2

        return ∇Ql, ∇ql, ∇Gl, ∇hl

    end function
    ```
