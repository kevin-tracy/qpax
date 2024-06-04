# pylint: disable=invalid-name
"""test.
here
"""
import jax 
import jax.numpy as jnp
import jax.scipy as jsp 


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

def optnet_derivatives(dz,dlam,dnu,z,lam,nu):

	dl_dQ = 0.5 * (jnp.outer(dz, z) + jnp.outer(z,dz)) 
	dl_dA = jnp.outer(dnu, z) + jnp.outer(nu, dz)
	dl_dG = jnp.diag(lam) @ (jnp.outer(dlam, z) + jnp.outer(lam, dz)) # TODO 

	dl_dq = dz 
	dl_db = - dnu
	dl_dh = - lam * dlam 

	return dl_dQ, dl_dq, dl_dA, dl_db, dl_dG, dl_dh 

def diff_qp(Q,q,A,b,G,h,z,s,lam, nu, dl_dz):
	nz = len(q)
	ns = len(h)
	nnu = len(b)

	# solve using same KKT solving functions 
	P_inv_vec, L_H, L_F = factorize_kkt(Q, G, A, s, lam)

	# stilde = G @ z - h 
	dz, ds, dlam_tilde, dnu = solve_kkt_rhs(Q,G,A,s,lam,P_inv_vec, L_H, L_F, -dl_dz, jnp.zeros(ns), jnp.zeros(ns), jnp.zeros(nnu))

	# recover real dlam from our modified (symmetrized) KKT system 
	dlam = dlam_tilde / lam 

	return optnet_derivatives(dz,dlam,dnu,z,lam,nu)

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


def pdip_step(inputs):

	# unpack inputs 
	Q, q, A, b, G, h, x, s, z, y, pdip_iter = inputs 

	# residuals 
	r1 = Q @ x + q + A.T @ y + G.T @ z 
	r2 = s * z 
	r3 = G @ x + s - h 
	r4 = A @ x - b

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

	return (Q, q, A, b, G, h, x, s, z, y, pdip_iter + 1)


def solve_qp(Q,q,A,b,G,h, max_iters=20, tol=1e-4):

	Q = 0.5*(Q + Q.T)
	
	x, s, z, y = initialize(Q,q,A,b,G,h)

	def continuation_criteria(inputs):
		Q, q, A, b, G, h, x, s, z, y, pdip_iter = inputs 

		duality_gap = s.dot(z)/len(s)
		ineq_res = jnp.linalg.norm(G @ x + s - h, ord = jnp.inf)

		return jnp.logical_and(pdip_iter < max_iters, jnp.logical_or(duality_gap > tol, ineq_res > tol))


	init_inputs = (Q, q, A, b, G, h, x, s, z, y, 0)

	outputs = jax.lax.while_loop(continuation_criteria, pdip_step, init_inputs)

	x, s, z, y, iters = outputs[6:11]

	return x, s, z, y, iters



"""
Here we define a function for solving the QP and returning the primal solution.

The implicit function theorem is used to comput these derivatives, and they 
are efficiently combined in a backwards pass with the tricks from OptNet. 

https://arxiv.org/abs/1703.00443
"""

@jax.custom_vjp
def solve_qp_primal(Q,q,A,b,G,h):
	return solve_qp(Q,q,A,b,G,h)[0]

def solve_qp_primal_fwd(Q,q,A,b,G,h):
	x,s,z,y,_ = solve_qp(Q,q,A,b,G,h)
	return x, (Q,q,A,b,G,h,x,s,z,y)

def solve_qp_primal_bwd(res, g):
	Q,q,A,b,G,h,x,s,z,y = res 
	return diff_qp(Q,q,A,b,G,h,x,s,z, y, g)

solve_qp_primal.defvjp(solve_qp_primal_fwd, solve_qp_primal_bwd)



"""
Here we define a function for solving the QP and returning the objective value.

Here we just take the gradient of the lagrangian at the primal-dual solution 
wrt the problem matrices to get the gradients of the objective value wrt
the problem matrices. This trick was shown in 2005 in 

"A closed formula for local sensitivity analysis in mathematical programming"
by Enrique Castillo et al 

https://www.researchgate.net/publication/250794318_A_closed_formula_for_local_sensitivity_analysis_in_mathematical_programming
"""

@jax.custom_vjp
def solve_qp_obj(Q,q,A,b,G,h):
	x = solve_qp(Q,q,A,b,G,h)[0]
	return 0.5*jnp.dot(x, Q @ x) + jnp.dot(q, x)

def solve_qp_obj_fwd(Q,q,A,b,G,h):
	x,s,z,y,_ = solve_qp(Q,q,A,b,G,h)
	val = 0.5*jnp.dot(x, Q @ x) + jnp.dot(q, x)
	res = (x, s, z, y)
	return val, res

def solve_qp_obj_bwd(res, g):
	x,s,z,y = res 

	gQ = 0.5* jnp.outer(x, x)
	gc = x
	gA = jnp.outer(y, x)
	gb = -y
	gG = jnp.outer(z, x)
	gh = -z

	return (gQ * g, gc * g, gA * g, gb * g, gG * g, gh * g)


solve_qp_obj.defvjp(solve_qp_obj_fwd, solve_qp_obj_bwd)

