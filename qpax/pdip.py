# pylint: disable=invalid-name
"""test.
here
"""
import jax 
import jax.numpy as jnp
import jax.scipy as jsp 
# from jax import custom_vjp

def qr_solve(qr, rhs):
	return jsp.linalg.solve_triangular(qr[1], qr[0].T @ rhs)

def initialize(Q, q, A, b, G, h):
	H = Q + G.T @ G 
	L_H = jnp.linalg.qr(H)
	F = A @ qr_solve(L_H, A.T)
	L_F = jnp.linalg.qr(F)

	r1 = -q + G.T @ h
	y = qr_solve(L_F, A @ qr_solve(L_H,r1) - b)
	x = qr_solve(L_H, r1 - A.T @ y)
	z = G @ x - h

	alpha_p = -jnp.min(-z)

	s = jnp.where(alpha_p < 0, -z, -z + (1 + alpha_p))

	alpha_d = -jnp.min(z)

	z = jnp.where(alpha_d >= 0, z + (1 + alpha_d), z)

	return x, s, z, y


def factorize_kkt(Q, G, A, s, z):
	P_inv_vec = z / s 
	H = Q + G.T @ (G.T * P_inv_vec).T 
	L_H = jnp.linalg.qr(H)
	F = A @ qr_solve(L_H, A.T)
	L_F = jnp.linalg.qr(F)

	return P_inv_vec, L_H, L_F 


def solve_kkt_rhs(Q,G,A,s,z,P_inv_vec, L_H, L_F, v1, v2, v3, v4):
	r2 = v3 - v2 / z
	p1 = v1 + G.T @ (P_inv_vec * r2)

	dy = qr_solve(L_F, A @ qr_solve(L_H,p1) - v4)
	dx = qr_solve(L_H, p1 - A.T @ dy)
	ds = v3 - G @ dx
	dz = (v2 - z * ds) / s

	return dx, ds, dz, dy 

def ort_linesearch(x,dx):
	# maximum alpha <= 1 st x + alpha * dx >= 0 
	body = lambda _x, _dx: jnp.where(_dx<0, -_x/_dx, jnp.inf)
	batch = jax.vmap(body, in_axes = (0,0))
	return jnp.min(jnp.array([1.0, jnp.min(batch(x,dx))]))

def centering_params(s, z, ds_a, dz_a):
	# duality gap + cc term in predictor-corrector PDIP 
	mu = jnp.dot(s, z)/len(s)

	alpha = jnp.min(jnp.array([
		ort_linesearch(s, ds_a),
		ort_linesearch(z, dz_a)]
	))

	sigma = (jnp.dot(s + alpha * ds_a, z + alpha * dz_a)/jnp.dot(s, z))**3

	return sigma, mu 


def pdip_pc_step(inputs):

	# unpack inputs 
	Q, q, A, b, G, h, x, s, z, y, solver_tol, converged, pdip_iter = inputs
	
	# residuals 
	r1 = Q @ x + q + A.T @ y + G.T @ z 
	r2 = s * z 
	r3 = G @ x + s - h 
	r4 = A @ x - b
	
	# check convergence 
	kkt_res = jnp.concatenate((r1, r2, r3, r4))
	converged = jnp.where(jnp.linalg.norm(kkt_res, ord=jnp.inf) < solver_tol, 1, 0)

	# affine step 
	P_inv_vec, L_H, L_F = factorize_kkt(Q,G,A,s,z)
	_, ds_a, dz_a, _ = solve_kkt_rhs(Q,G,A,s,z,P_inv_vec, L_H, L_F, -r1, -r2, -r3, -r4)

	# change centering + correcting params 
	sigma, mu = centering_params(s, z, ds_a, dz_a)
	r2 = r2 - (sigma * mu - (ds_a * dz_a))
	dx, ds, dz, dy = solve_kkt_rhs(Q,G,A,s,z,P_inv_vec, L_H, L_F, -r1, -r2, -r3, -r4)

	# linesearch and update primal & dual vars 
	alpha = 0.99*jnp.min(jnp.array([
				ort_linesearch(s, ds),
				ort_linesearch(z, dz) 
				]))

	x = x + alpha * dx 
	s = s + alpha * ds 
	z = z + alpha * dz 
	y = y + alpha * dy 

	return (Q, q, A, b, G, h, x, s, z, y, solver_tol,
	 converged, pdip_iter + 1)

# 0 Q
# 1 q
# 2 A
# 3 b
# 4 G
# 5 h
# 6 x
# 7 s
# 8 z
# 9 y
# 10 solver_tol
# 11 converged
# 12 pdip_iter

def solve_qp(Q,q,A,b,G,h, solver_tol=1e-3):

	# symmetrize Q 
	Q = 0.5*(Q + Q.T)

	x, s, z, y = initialize(Q,q,A,b,G,h)

	# continuation criteria for normal predictor-corrector
	def pc_continuation_criteria(inputs):

		converged = inputs[11]
		pdip_iter = inputs[12]

		return jnp.logical_and(pdip_iter < 30, converged == 0)


	converged = 0
	pdip_iter = 0
	init_inputs = (Q, q, A, b, G, h, x, s, z, y, solver_tol, converged, pdip_iter)

	outputs = jax.lax.while_loop(pc_continuation_criteria, pdip_pc_step, init_inputs)

	x, s, z, y = outputs[6:10]
	converged = outputs[11]
	pdip_iter = outputs[12]

	return x, s, z, y, converged, pdip_iter


def pdip_pc_step_debug(inputs):

	# unpack inputs 
	Q, q, A, b, G, h, x, s, z, y, solver_tol, converged, pdip_iter = inputs
	
	# residuals 
	r1 = Q @ x + q + A.T @ y + G.T @ z 
	r2 = s * z 
	r3 = G @ x + s - h 
	r4 = A @ x - b
	
	# check convergence 
	kkt_res = jnp.concatenate((r1, r2, r3, r4))
	converged = jnp.where(jnp.linalg.norm(kkt_res, ord=jnp.inf) < solver_tol, 1, 0)

	# affine step 
	P_inv_vec, L_H, L_F = factorize_kkt(Q,G,A,s,z)
	_, ds_a, dz_a, _ = solve_kkt_rhs(Q,G,A,s,z,P_inv_vec, L_H, L_F, -r1, -r2, -r3, -r4)

	# change centering + correcting params 
	sigma, mu = centering_params(s, z, ds_a, dz_a)
	r2 = r2 - (sigma * mu - (ds_a * dz_a))
	dx, ds, dz, dy = solve_kkt_rhs(Q,G,A,s,z,P_inv_vec, L_H, L_F, -r1, -r2, -r3, -r4)

	# linesearch and update primal & dual vars 
	alpha = 0.99*jnp.min(jnp.array([
                1.0,
                .99 * ort_linesearch(s, ds),
                .99 * ort_linesearch(z, dz) 
                ]))

	x = x + alpha * dx 
	s = s + alpha * ds 
	z = z + alpha * dz 
	y = y + alpha * dy 

	if len(r4) == 0:
		r4 = jnp.zeros(1)
	print("%3d   %9.2e   %9.2e  %9.2e  %9.2e   % 6.4f" %
			(
				pdip_iter,
				jnp.linalg.norm(r1, ord=jnp.inf),
				jnp.linalg.norm(r2, ord=jnp.inf),
				jnp.linalg.norm(r3, ord=jnp.inf),
				jnp.linalg.norm(r4, ord=jnp.inf),
				alpha
			)
	)  

	return (Q, q, A, b, G, h, x, s, z, y, solver_tol,
	 converged, pdip_iter + 1)


def while_loop_debug(cond_fun, body_fun, init_val):
	val = init_val 
	while cond_fun(val):
		val = body_fun(val)
	return val 

def solve_qp_debug(Q,q,A,b,G,h, solver_tol=1e-3):

	# symmetrize Q 
	Q = 0.5*(Q + Q.T)

	x, s, z, y = initialize(Q,q,A,b,G,h)

	# continuation criteria for normal predictor-corrector
	def pc_continuation_criteria(inputs):

		converged = inputs[11]
		pdip_iter = inputs[12]

		return jnp.logical_and(pdip_iter < 30, converged == 0)


	converged = 0
	pdip_iter = 0
	init_inputs = (Q, q, A, b, G, h, x, s, z, y, solver_tol, converged, pdip_iter)
	
	print("iter      r1          r2         r3         r4        alpha")
	print("------------------------------------------------------")

	outputs = while_loop_debug(pc_continuation_criteria, pdip_pc_step_debug, init_inputs)

	x, s, z, y = outputs[6:10]
	converged = outputs[11]
	pdip_iter = outputs[12]

	return x, s, z, y, converged, pdip_iter