# qpax
Differentiable QP solver in [JAX](https://github.com/google/jax).

This package can be used for solving convex quadratic programs of the following form:

$$
\begin{align*}
\underset{x}{\text{minimize}} & \quad \frac{1}{2}x^TQx + q^Tx \\
\text{subject to} & \quad  Ax = b, \\
                  & \quad  Gx \leq h
\end{align*}
$$

where $Q \succeq 0$. This solver can be combined with JAX's `jit` and `vmap` functionality, as well as differentiated with reverse-mode `grad`. 

The QP is solved with a primal-dual interior point algorithmn detailed in [cvxgen](https://stanford.edu/~boyd/papers/pdf/code_gen_impl.pdf), with the solution to the linear systems computed with reduction techniques from [cvxopt](http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf). At an approximate primal-dual solution, the the primal variable $x$ is differentiated with respect to the problem parameters using the implicit function theorem as shown in [optnet](https://arxiv.org/abs/1703.00443), and their pytorch qp solver [qpth](https://github.com/locuslab/qpth).

## Installation

To install directly from github using `pip`:

```bash
$ pip install git+https://github.com/kevin-tracy/qpax
```

Alternatively, to install from source:

```bash
$ python setup.py install
```

## Usage

### Solving a QP 
We can solve QP's with qpax in a way that plays nice with JAX's `jit` and `vmap`:
```python 
ipmort qpax

# solve QP (this can be combined with jit or vmap)
x, s, z, y, iters = qpax.solve_qp(Q, q, A, b, G, h)
```
### Solving a batch of QP's 

Here let's solve a batch of nonnegative least squares problems as QPs. This outlines two bits of functionality from `qpax`, first is the ability to solve QP's without any equality constraints, and second is the ability to `vmap` over a batch of QP's. 

```python 
import numpy as np
import jax 
import jax.numpy as jnp 
from jax import jit, grad, vmap  
import qpax 

# non-negative least squares size
n = 5   # size of x 
m = 10  # rows in A 

# create data for N_qps random qp's 
N_qps = 100 

# cost terms 
Qs = jnp.zeros((N_qps, n, n))
qs = jnp.zeros((N_qps, n))

# these stay as zero when there are no equality constraints 
As = jnp.zeros((N_qps, 0, n))
bs = jnp.zeros((N_qps, 0))

# inequality terms 
Gs = jnp.zeros((N_qps, n, n))
hs = jnp.zeros((N_qps, n))

for i in range(N_qps):
  A = jnp.array(np.random.randn(m,n))
  b = jnp.array(np.random.randn(m))
  
  # least squares objective terms 
  Qs = Qs.at[i].set(A.T @ A)
  qs = qs.at[i].set(-A.T @ b)
  
  # nonnegative constraint
  Gs = Gs.at[i].set(-jnp.eye(n))

# vmap over this 
batch_qp = jit(vmap(qpax.solve_qp_x, in_axes = (0,0,0,0,0,0)))

xs = batch_qp(Qs, qs, As, bs, Gs, hs)
```

### Differentiating a QP 

Alternatively, if we are only looking to use the primal variable `x`, we can use `solve_qp_x` and enable automatic differenation:

```python
import jax 
import jax.numpy as jnp 
import qpax 

def loss(Q, q, A, b, G, h):
    x = qpax.solve_qp_x(Q, q, A, b, G, h) 
    x_bar = jnp.ones(len(q))
    return jnp.dot(x - x_bar, x - x_bar)
  
# gradient of loss function   
loss_grad = jax.grad(loss, argnums = (0, 1, 2, 3, 4, 5))

# compatible with jit 
loss_grad_jit = jax.jit(loss_grad)

# calculate derivatives 
derivs = loss_grad_jit(Q, q, A, b, G, h)
dl_dQ, dl_dq, dl_dA, dl_db, dl_dG, dl_dh = derivs 
```
