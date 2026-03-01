
import meshio
import numpy as np
from helper import *
import jax.numpy as jnp
import jax
import time
import jax.lax as lax
import os
np.set_printoptions(precision=5)
jax.clear_caches()
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
np.set_printoptions(precision=5) 

#%load_ext line_profiler
jax.config.update('jax_default_matmul_precision', 'highest') 
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)

cache_path = os.path.abspath("./jax_cache")
if not os.path.exists(cache_path):
    os.makedirs(cache_path)
cache_path = os.path.abspath("./jax_cache")
if not os.path.exists(cache_path):
    os.makedirs(cache_path)


os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.80'
os.environ['JAX_COMPILATION_CACHE_DIR'] = cache_path
os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'

start = time.time()
#print(f"  Start Compute: {start:.4f}s")
############### User Input #################################
name='Sandwich_SC_files/RHC/RHC_refined_45'
#name='Sandwich_SC_files/RHC/SW_2UC_45' 
#name='Sandwich_SC_files/RHC/RHC_Coarse_2UC_45'
#name='Sandwich_SC_files/BCC/SW_2UC_45'

material_param=jnp.array([(108e3,8e3,8e3,4e3,4e3,3e3,0.32,0.32,0.30),
                          (108e3,8e3,8e3,4e3,4e3,3e3,0.32,0.32,0.30),
                         (69e3, 69e3, 69e3, 26.54e3, 26.54e3, 26.54e3, 0.30, 0.30, 0.30)])

angles = jnp.array([45, -45, 0.0]) # Put 0.0 if no angle used
#print('JAX initial', time.time()- start)
#####################################################

num_sg=generate_msh_from_sc(name+'.sc', 'sgmesh.msh')
#num_sg=3
mesh = meshio.read('sgmesh.msh') 
points = jnp.array(mesh.points, dtype=np.float32)[:,0:num_sg]
cells = jnp.array(mesh.cells[0].data, dtype=np.uint64) 

meshio_type = mesh.cells[0].type 
cell_type_map = {
    "triangle": CellType.triangle,
    "quad": CellType.quadrilateral,
    "tetra": CellType.tetrahedron,
    "hexahedron": CellType.hexahedron
}

fe_type = FiniteElementType(
    cell_type=cell_type_map.get(meshio_type),
    family=ElementFamily.P,
    basis_degree=1,
    lagrange_variant=LagrangeVariant.equispaced,
    quadrature_type=QuadratureType.default,
    quadrature_degree=3,
)

def build_single_C_matrix(params):
    """Helper function to build a 6x6 C matrix from a 1D array of 9 properties."""
    E1, E2, E3, G12, G13, G23, v12, v13, v23 = params
    
    # Initialize empty 6x6 matrix
    S = jnp.zeros((6, 6))
    
    S = S.at[0, 0].set(1/E1)
    S = S.at[1, 1].set(1/E2)
    S = S.at[2, 2].set(1/E3)
    S = S.at[3, 3].set(1/G23)
    S = S.at[4, 4].set(1/G13)
    S = S.at[5, 5].set(1/G12)
    
    S = S.at[0, 1].set(-v12/E1).at[1, 0].set(-v12/E1)
    S = S.at[0, 2].set(-v13/E1).at[2, 0].set(-v13/E1)
    S = S.at[1, 2].set(-v23/E2).at[2, 1].set(-v23/E2)
    C= jnp.linalg.inv(S)
    return C

def rotate_C_matrix(C, t):
    """Rotates a 6x6 stiffness matrix C by angle t. Skips math if t=0."""

    def do_rotation(operand):
        C_mat, angle = operand
        th = jnp.deg2rad(angle)
        c, s = jnp.cos(th), jnp.sin(th)
        cs = c * s
        # theta ccw from from 123 direction (fiber/material)
        #Exam 3.1 of MSM book with theta ccw.
        R_Sig = jnp.array([
            [c**2, s**2, 0, 0, 0, 2*cs],
            [s**2, c**2, 0, 0, 0, -2*cs],
            [0,    0,    1, 0, 0, 0    ],
            [0,    0,    0, c, -s, 0    ],
            [0,    0,    0, s, c, 0   ],
            [-cs,  cs,   0, 0, 0, c**2 - s**2]
        ])
        return R_Sig @ C_mat @ R_Sig.T

    def skip_rotation(operand):
        C_mat, _ = operand
        return C_mat

    # jax.lax.cond(condition, true_function, false_function, arguments)
    return jax.lax.cond(
        t == 0.0,         # Condition
        skip_rotation,    # Run this if True
        do_rotation,      # Run this if False
        (C, t)            # Pack the arguments to pass into the functions
    )

cell_domain_ids = jnp.array(mesh.cell_data["gmsh:physical"][0]-1)   
@partial(jax.jit, static_argnames=['num_quad_points'])
def get_heterogeneous_C_matrix(cell_domain_ids, num_quad_points, material_param, domain_angles):

    C_matrices_base = jax.vmap(build_single_C_matrix)(material_param)
    
    C_matrices_rotated = jax.vmap(rotate_C_matrix)(C_matrices_base, domain_angles)
    
    return C_matrices_rotated[cell_domain_ids]

@jax.jit    
def _element_residual_single_case(
    u_nd: jnp.ndarray,
    x_nd: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    phi_qn: jnp.ndarray,
    W_q: jnp.ndarray,
    C_ss: jnp.ndarray,
    epsilon_bar: jnp.ndarray = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
):
    J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
    G_qpd = jnp.linalg.inv(J_qdp)
    det_J_q = jnp.linalg.det(J_qdp)
    dphi_dx_qnd = jnp.einsum("qpd,qnp->qnd", G_qpd, dphi_dxi_qnp)
    
    Q, N, _ = dphi_dx_qnd.shape
    zeros = jnp.zeros((Q, N), dtype=dphi_dx_qnd.dtype)
    if x_nd.shape[1]==1: # 1D SG
        dphi_dx_3D = jnp.stack([zeros, zeros, dphi_dx_qnd[..., 0]], axis=-1)
    elif x_nd.shape[1]==2: # 2D SG
        dphi_dx_3D = jnp.stack([zeros, dphi_dx_qnd[..., 0], dphi_dx_qnd[..., 1]], axis=-1)
    elif x_nd.shape[1]==3: # 3D SG
        dphi_dx_3D = dphi_dx_qnd

    dx = dphi_dx_3D[..., 0]
    dy = dphi_dx_3D[..., 1]
    dz = dphi_dx_3D[..., 2]
    
    u_x, u_y, u_z = u_nd[:, 0], u_nd[:, 1], u_nd[:, 2]
    eps_xx = dx @ u_x
    eps_yy = dy @ u_y
    eps_zz = dz @ u_z
    eps_yz = (dy @ u_z) + (dz @ u_y)
    eps_xz = (dx @ u_z) + (dz @ u_x)
    eps_xy = (dx @ u_y) + (dy @ u_x)

    eps_qs = jnp.stack([eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy], axis=-1) + epsilon_bar
    stress = jnp.einsum("ij, qj, q, q -> qi", C_ss, eps_qs, det_J_q, W_q) 

    R_x = jnp.sum(stress[:, 0, None] * dx + stress[:, 5, None] * dy + stress[:, 4, None] * dz, axis=0)
    R_y = jnp.sum(stress[:, 5, None] * dx + stress[:, 1, None] * dy + stress[:, 3, None] * dz, axis=0)
    R_z = jnp.sum(stress[:, 4, None] * dx + stress[:, 3, None] * dy + stress[:, 2, None] * dz, axis=0)

    R_nd = jnp.stack([R_x, R_y, R_z], axis=-1)
    
    f_vol_scalar = jnp.einsum("qn, q, q -> n", phi_qn, det_J_q, W_q)
    f_vol = jnp.tile(f_vol_scalar[:, None], (1, 3))
    return R_nd, f_vol

@jax.jit
def calculate_residual_batch_element_kernel_mixed_periodic(
    x_end: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    phi_qn: jnp.ndarray,          
    W_q: jnp.ndarray,
    C_ess: jnp.ndarray,
    periodic_cells, # Use the periodic map!
    unique_dofs_full: jnp.ndarray, # Pass your unique indices (including Lagrange)
    u_g_flat: jnp.ndarray,   
):
    #u_g_flat, lamb_vec = u_red[:-3], u_red[-3:]
    
    #E=x_end.shape[0]
    u_end = transform_global_unraveled_to_element_node(periodic_cells, u_g_flat, U=3)
    def element_process_all_cases(u_nd, x_nd, C_ss):
        run_6_cases = jax.vmap( # Inner VMAP: Iterate over the 6 unit strain vectors 
            _element_residual_single_case, 
            in_axes=(None, None, None, None, None, None, 1)
        )
        R_6_nodes_3 = run_6_cases(
            u_nd, x_nd, dphi_dxi_qnp, phi_qn, W_q, C_ss, jnp.eye(6)
        )[0]
        return R_6_nodes_3
        
    batch_processor = jax.vmap(  # 3. Outer VMAP: Iterate over the batch of elements
        element_process_all_cases, 
        in_axes=(0, 0, 0)
    )
    
    R_end_cases = batch_processor(u_end, x_end, C_ess) # (E,6,N,3)
    R_cases_first = jnp.transpose(R_end_cases, (1, 0, 2, 3))
    def assemble_one_case(R_one_case_EN3):  #
        return transform_element_node_to_global_unraveled_sum(#(GlobalNDOF_unraveled, 3) 
            periodic_cells=periodic_cells, 
            v_en=R_one_case_EN3,
            num_nodes=u_g_flat.shape[0]//3,
        )
    R_global_cases = jax.vmap(assemble_one_case)(R_cases_first)
    R_elastic_matrix = R_global_cases.reshape(6, -1).T
    R_elastic_matrix = R_elastic_matrix.at[0:3, :].set(0.0)
   # constraint_rhs = jnp.zeros((3, 6))
   # R_mixed_matrix = jnp.concatenate([R_elastic_matrix, constraint_rhs], axis=0)
    return R_elastic_matrix

@jax.jit
def _calculate_jacobian_batch_element_kernel_periodic(
    x_end: jnp.ndarray,
    u_f,
    dphi_dxi_qnp: jnp.ndarray,
    phi_qn: jnp.ndarray,          
    W_q: jnp.ndarray,
    C_ess: jnp.ndarray,
    periodic_cells: object,
):
    E = x_end.shape[0]
   # u_f = u_red[:-3]
   # lamb_f = u_red[-3:] 
    u_enu = transform_global_unraveled_to_element_node(periodic_cells, u_f)

    N = x_end.shape[1]
    D = x_end.shape[2]
    U = u_enu.shape[2]
    u_et = u_enu.reshape(E, N * U)
    #lamb_et = jnp.tile(lamb_f, (E, 1)) 
    #u_mixed_et = jnp.concatenate([u_et, lamb_et], axis=1) # Shape: (E, 12)
    
    @jax.jit
    def residual_kernel(u_t, x_nd, C_ss):
      u_nd = u_t.reshape(N, U)
      R_elastic, _ = _element_residual_single_case(
          u_nd, x_nd, dphi_dxi_qnp, phi_qn, W_q, C_ss, 
          epsilon_bar=jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      )
      return R_elastic.ravel() # Pure Kuu matrix!

    J_uu = jax.vmap(jax.jacfwd(residual_kernel, argnums=0))(
        u_et, x_end, C_ess
    )

    return J_uu

def ebe_jacobian_product_periodic(
    J_uu, periodic_cells: jnp.ndarray, n_unique_u: int, z_u: jnp.ndarray
):
    z_u_enu = transform_global_unraveled_to_element_node(
        periodic_cells, z_u
    )
    _,N,U=z_u_enu.shape
    z_u_local = z_u_enu.reshape(-1, N*U)

    f_u_local_flat = jnp.einsum('eij,ej->ei', J_uu, z_u_local) 
    f_u_local = f_u_local_flat.reshape(-1, N, U)
                     
    f_u_global_full = transform_element_node_to_global_unraveled_sum(
        periodic_cells=periodic_cells, v_en=f_u_local, num_nodes=n_unique_u // U,
    )
    f_u_global_full = f_u_global_full.at[0:3].set(z_u[0:3])
    return f_u_global_full

# --- Define the Block Preconditioner Apply Function ---
@partial(jax.jit, static_argnames=["n_unique_u"])
def apply_block_precond(inv_blocks, n_unique_u, x_reduced):
    
    # Apply 3x3 block matrix multiplication to each node
    x_nodes = x_reduced.reshape(-1, 3)
    precond_nodes = jnp.einsum('nij,nj->ni', inv_blocks, x_nodes)
    
    return precond_nodes.ravel()
    
@partial(jax.jit, static_argnames=["n_unique"])
def compute_block_inv_diag(J_uu, periodic_cells, n_unique):
    E, n_u, _ = J_uu.shape
    N = n_u // 3
    
    # 1. Extract 3x3 diagonal blocks for each node in each element
    # Reshape to (E, N, 3, N, 3) to isolate the nodal blocks
    J_uu_reshaped = J_uu.reshape(E, N, 3, N, 3)
    
    # Extract blocks where the node index matches. Shape becomes (E, 3, 3, N)
    local_3x3_blocks = jnp.diagonal(J_uu_reshaped, axis1=1, axis2=3) 
    
    # Rearrange to (E, N, 3, 3)
    local_3x3_blocks = jnp.moveaxis(local_3x3_blocks, -1, 1) 
    
    # 2. Scatter-add to form the global nodal 3x3 matrices
    num_unique = n_unique // 3
    global_blocks = jnp.zeros((num_unique, 3, 3))
    global_blocks = global_blocks.at[periodic_cells].add(local_3x3_blocks)
    
    # 3. Invert the 3x3 blocks directly on the GPU
    I_3x3 = jnp.eye(3)[None, :, :]
    global_blocks_safe = global_blocks + I_3x3 * 1e-8
    
    # jnp.linalg.inv automatically batches over the first dimension!
    inv_global_blocks = jnp.linalg.inv(global_blocks_safe) # Shape: (num_nodes, 3, 3)
    
    return inv_global_blocks

# 2. Power Method for Eigenvalue Estimation      
# 1. Remove @jax.jit here! lax.map already handles the compilation.
def estimate_max_eigenvalue(A_op, M_op, b_col, num_iters=15):
    """
    Uses the Power Method to find the maximum eigenvalue of M^-1 * A.
    """
    # Use ones_like to completely bypass shape tracing errors
    v = jnp.ones_like(b_col)
    v = v / jnp.linalg.norm(v)
    
    def power_step(i, state):
        v_current, current_lam = state
        
        w = M_op(A_op(v_current)) 
        lam_new = jnp.vdot(v_current, w)
        
        # Added a tiny epsilon (1e-12) to prevent division by zero just in case
        v_new = w / (jnp.linalg.norm(w) + 1e-12) 
        
        return (v_new, lam_new)
        
    _, lambda_max = jax.lax.fori_loop(0, num_iters, power_step, (v, 0.0))
    
    return lambda_max * 1.05

# 3. The Chebyshev Wrapper
def apply_chebyshev_precond(inv_blocks, n_unique_u, eig_max, eig_min, A_op, degree, x_reduced):
    """
    Notice x_reduced is the LAST argument. 
    BiCGSTAB will pass the current residual into this slot.
    """
    d = (eig_max + eig_min) / 2.0
    c = (eig_max - eig_min) / 2.0
    
    i = jnp.arange(1, degree + 1)
    theta = d + c * jnp.cos(jnp.pi * (2 * i - 1) / (2 * degree))
    
    def step(z, theta_i):
        # A_op(z) expects a vector
        r = x_reduced - A_op(z) 
        
        # Here we manually pass 'r' into the x_reduced slot of your base preconditioner
        z_update = apply_block_precond(inv_blocks, n_unique_u, r)
        
        z_new = z + z_update / theta_i
        return z_new, None

    z0 = jnp.zeros_like(x_reduced)
    z_final, _ = jax.lax.scan(step, z0, theta)
    return z_final

# 4. Compute D1
def compute_effective_properties(
    delta_u_matrix: jnp.ndarray,
    R_f_reduced: jnp.ndarray,
    x_end: jnp.ndarray,        
    dphi_dxi_qnp: jnp.ndarray, 
    W_q: jnp.ndarray,          
    C_ess: jnp.ndarray         
):
    # 1. Compute D1 instantly
    D1 = jnp.einsum('ni,nj->ij', delta_u_matrix, R_f_reduced)

    # 2. Extract volumes (vmap over elements)
    def _get_vol(x_nd):
        J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
        return jnp.sum(jnp.linalg.det(J_qdp) * W_q)
    
    elem_vols = jax.vmap(_get_vol)(x_end)
    omega = jnp.sum(elem_vols)

    # 3. Compute D_bar with ZERO memory bloat
    # Multiplies (E, 6, 6) by (E,) and sums across elements instantly
    D_bar = jnp.einsum('eij,e->ij', C_ess, elem_vols)

    # 4. Effective Stiffness
    C_eff = (D_bar + D1) / omega

    # 5. Invert for Compliance (jnp.linalg.inv is fine for 6x6)
    I = jnp.eye(6, dtype=C_eff.dtype)
    Com = jnp.linalg.solve(C_eff, I)
  
    # 6. Extract Engineering Constants on the GPU
    E1, E2, E3 = 1/Com[0,0], 1/Com[1,1], 1/Com[2,2]
    v12, v13, v23 = -Com[0,1]/Com[0,0], -Com[0,2]/Com[0,0], -Com[1,2]/Com[1,1]
    G23, G13, G12 = 1/Com[3,3], 1/Com[4,4], 1/Com[5,5]

    return C_eff, jnp.array([E1, E2, E3, G12, G13, G23,v12, v13, v23])

#########################################################################
@partial(jax.jit, static_argnames=['n_unique'])
def full_homogenization_pipeline(
    x_end, u_0_g, dphi_dxi_qnp, phi_qn, W_q, 
    periodic_cells, unique_dofs, n_unique
):
    C_ess = get_heterogeneous_C_matrix(cell_domain_ids, num_quad_points=Q, 
    material_param=material_param, domain_angles=angles
    )

    Dhe = -calculate_residual_batch_element_kernel_mixed_periodic(
        x_end, dphi_dxi_qnp, phi_qn, W_q, C_ess, periodic_cells, 
        unique_dofs, u_0_g_full[unique_dofs]
    )

    J_uu = _calculate_jacobian_batch_element_kernel_periodic(
        x_end, u_0_g_full[unique_dofs], dphi_dxi_qnp, phi_qn, W_q, C_ess, periodic_cells
    )
     
    # 3. Solver Prep
    #t3 = time.time()
    inv_blocks = compute_block_inv_diag(J_uu, periodic_cells, n_unique)
   # print(f"  Preconditioner time: {time.time()-start:.4f}s")
 
    def solve_inner(b_col):
        # A_op expects ONE argument: the vector to multiply
        A_op = jax.tree_util.Partial(
            ebe_jacobian_product_periodic, 
            J_uu, periodic_cells, n_unique
        )
        
        # M_op freezes the first 3 args, waiting for 'x_reduced'
        M_op = jax.tree_util.Partial(
            apply_block_precond, 
            inv_blocks, n_unique
        )
        # Estimate the max eigenvalue dynamically using our M_op and A_op
        estimated_eig_max = estimate_max_eigenvalue(A_op, M_op, b_col, num_iters=15)
        estimated_eig_min = estimated_eig_max / 25.0 
        
        # Freeze all arguments EXCEPT x_reduced
        cheb_M = jax.tree_util.Partial(
            apply_chebyshev_precond,
            inv_blocks,     # arg 
            n_unique,       # arg 
            estimated_eig_max, # arg 
            estimated_eig_min, # arg
            A_op,           # arg 
            4           
        )
        res, _ = jax.scipy.sparse.linalg.cg(A_op, b_col,M=cheb_M, tol=1e-8, atol=1e-8,
        maxiter=2000)
        return res

    delta_u_matrix = jax.lax.map(solve_inner, Dhe.T).T

    D_eff, constants = compute_effective_properties(
        delta_u_matrix, -Dhe, x_end, dphi_dxi_qnp, W_q, C_ess
    )
   # N_nodes = n_unique // 3
   # delta_u_3d = delta_u_matrix.reshape(N_nodes, 3, 6)
   # mean_u = jnp.mean(delta_u_3d, axis=0, keepdims=True)
   # delta_u_zero_mean = (delta_u_3d - mean_u).reshape(n_unique, 6)
    print(f"  D_eff process: {time.time()-start:.4f}s")
    return D_eff, constants

##########################################################################

xi_qp, W_q = get_quadrature(fe_type=fe_type)
phi_qn, dphi_dxi_qnp = eval_basis_and_derivatives(fe_type=fe_type, xi_qp=xi_qp)
Q = xi_qp.shape[0] 
V=points.shape[0]
U=3 # num of solution components

x_end = mesh_to_jax(vertices=points, cells=cells)

E = x_end.shape[0] # num elements
print(f"  Quadrature i/p: {time.time()-start:.4f}s")    

# Periodic assembly map input

periodic_cells, dof_map_np = mesh_to_periodic_sparse_assembly_map(V, cells, points,tol=1e-6)
unique_dofs = jnp.unique(dof_map_np)
n_unique=len(unique_dofs)
u_0_g_full = jnp.zeros(shape=(V * U)) 

C_eff, props = full_homogenization_pipeline(
    x_end, u_0_g_full, dphi_dxi_qnp, phi_qn, W_q,  
    periodic_cells, unique_dofs, n_unique
)    
########################################################################

labels = ["E1", "E2", "E3", "G12", "G13", "G23", "v12", "v13", "v23"];
print("--- Effective Material Properties ---")
for label, val in zip(labels, props):
    print(f"{label}: {val}")
print('\n Effective Stiffness matrix \n')
print(C_eff)
print(f" \n Total Time taken: {time.time()-start:.4f}s")  
