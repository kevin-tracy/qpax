
import jax 
import jax.numpy as jnp
import jax.scipy as jsp 
# from jax import grad 
# from jax import jacobian 
from jax import jit 
# from jax import vmap 
# import numpy as np 
# from jax import custom_vjp


def qr_solve(qr, rhs):
	return jsp.linalg.solve_triangular(qr[1], qr[0].T @ rhs)


def initialize(Q,q,A,b,G,h):
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

@jit 
def solve_qp(Q,q,A,b,G,h):

	Q = 0.5*(Q + Q.T)
	
	x, s, z, y = initialize(Q,q,A,b,G,h)

	def continuation_criteria(inputs):
		Q, q, A, b, G, h, x, s, z, y, pdip_iter = inputs 

		duality_gap = s.dot(z)/len(s)
		ineq_res = jnp.linalg.norm(G @ x + s - h, ord = jnp.inf)

		return jnp.logical_and(pdip_iter < 20, jnp.logical_or(duality_gap > 1e-4, ineq_res > 1e-4))


	init_inputs = (Q, q, A, b, G, h, x, s, z, y, 0)

	outputs = jax.lax.while_loop(continuation_criteria, pdip_step, init_inputs)

	x, s, z, y, iters = outputs[6:11]

	return x, s, z, y, iters

@jit 
def my_f(a):
	return 4 * a 
# nx = 15
# ns = 10
# nz = ns 
# ny = 3 
# Q, q, A, b, G, h, x_true, s_true, z_true, y_true = gen_qp(nx, ns, ny)


# x,s,z,y,iters = solve_qp(Q,q,A,b,G,h)

# print("solution x: ", x)

# print(jnp.linalg.norm(x - x_true))
# print(jnp.linalg.norm(s - s_true))


# check = lambda var: print("variable type: ", var.dtype, "weak type: ", var.weak_type)

# check(x)
# check(s)
# check(z)
# check(y)


# print(jnp.linalg.norm(z - z_true))
# print(jnp.linalg.norm(y - y_true))
# r1 = Q @ x + q + A.T @ y + G.T @ z 
# r2 = s * z 
# r3 = G @ x + s - h 
# r4 = A @ x - b

# print(jnp.linalg.norm(r1))
# print(jnp.linalg.norm(r2))
# print(jnp.linalg.norm(r3))
# print(jnp.linalg.norm(r4))
# print("iters: ", iters)


# @jax.custom_vjp
# def solve_qp_x(Q,q,A,b,G,h):
# 	return solve_qp(Q,q,A,b,G,h)[0]

# @jit 
# def solve_qp_forward(Q,q,A,b,G,h):
# 	x,s,z,y,_ = solve_qp(Q,q,A,b,G,h)
# 	return x, (Q,q,A,b,G,h,x,s,z,y)

# @jit 
# def solve_qp_backward(res, g):
# 	Q,q,A,b,G,h,x,s,z,y = res 
# 	return diff_qp(Q,q,A,b,G,h,x,s,z, y, g)

# solve_qp_x.defvjp(solve_qp_forward, solve_qp_backward)

# solve_qp_x = jit(solve_qp_x)



# def my_f(Q,q,A,b,G,h):
# 	x = solve_qp_x(Q,q,A,b,G,h) 
# 	x_bar = jnp.ones(len(q))
# 	return jnp.dot(x - x_bar, x-x_bar)


# def vectorize_mat(mat):
# 	r,c = mat.shape 
# 	return jnp.reshape(mat, (r*c,))

# def materize_vec(vec, dims):
# 	return jnp.reshape(vec, dims)


# def finite_difference(func, x):
# 	y = func(x)
# 	dx = 1e-3
# 	# finite diff of a vector 
# 	if len(x.shape) == 1: 
	
# 		g = jnp.zeros(len(x))

		

# 		for i in range(len(x)):

# 			x2 = x.at[i].set(x[i] + dx)
# 			y2 = func(x2)

# 			g = g.at[i].set((y2 - y)/dx)

# 		return g 
# 	else:

# 		nr,nc = x.shape
# 		nel = nr * nc 
# 		g = jnp.zeros(nel)

# 		for i in range(nel):

# 			x2 = vectorize_mat(x)
# 			x2 = x2.at[i].set(x2[i] + dx)

# 			y2 = func(materize_vec(x2, (nr,nc)))

# 			g = g.at[i].set((y2 - y)/dx)


# 		return materize_vec(g, (nr, nc))



# inputs = (Q,q,A,b,G,h)

# print("here is my f: ", my_f(*inputs))

# derivs = grad(my_f, argnums = (0,1,2,3,4,5))(*inputs)


# def my_f_select(inputs, X, i):
# 	# replace the ith element of inputs with X 
# 	new_inputs = tuple(X if index == i else value for index, value in enumerate(inputs))
# 	return my_f(*new_inputs)


# def filter_out_small(A):
# 	A = np.array(A)

# 	threshold = 1e-3 

# 	mask = np.abs(A) < threshold 

# 	A[mask] = 0 

# 	return jnp.array(A)

# input_names = ("Q","q","A","b","G","h")
# for i in range(6):
# 	print("-------------input: ", input_names[i], "----------------")

# 	lambda_f = lambda _X: my_f_select(inputs, _X, i)
# 	fd_deriv = finite_difference(lambda_f, inputs[i])

# 	assert fd_deriv.shape == derivs[i].shape

# 	# print("deriv: ")
# 	# print(filter_out_small(derivs[i]))
# 	print("fd_deriv_norm: ")
# 	print(jnp.linalg.norm(fd_deriv))
# 	print("error_norm: ", jnp.linalg.norm(derivs[i] - fd_deriv))




