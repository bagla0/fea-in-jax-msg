import meshio
import numpy as np
from helper import *
#import pyvista
import jax.numpy as jnp
import jax
import time
np.set_printoptions(precision=5)

#%load_ext line_profiler
jax.config.update('jax_default_matmul_precision', 'highest') 
jax.config.update("jax_enable_x64", True)

start = time.time()
############### User Input #################################

path='Sandwich_SC_files/HC/HC_2UC_SW_45'
material_param=jnp.array([(108e3,8e3,8e3,4e3,4e3,3e3,0.32,0.32,0.30),
                          (108e3,8e3,8e3,4e3,4e3,3e3,0.32,0.32,0.30),
                         (69e3, 69e3, 69e3, 26.54e3, 26.54e3, 26.54e3, 0.30, 0.30, 0.30)])

angles = jnp.array([45, -45, 0.0]) # Put 0.0 if no angle used

############################################
num_sg=generate_msh_from_sc(path+'.sc','sg_mesh.msh')
mesh = meshio.read('sg_mesh.msh') 
points = np.array(mesh.points, dtype=np.float32)[:,0:num_sg]
cells = np.array(mesh.cells[0].data, dtype=np.uint64) 

#pv_mesh = pyvista.from_meshio(mesh)
#pv_mesh.cell_data["Subdomains"] = mesh.cell_data["gmsh:physical"][0]
#plotter = pyvista.Plotter()
#plotter.add_mesh(pv_mesh, scalars="Subdomains",show_edges=True)
#plotter.add_axes()
#plotter.show()


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
    quadrature_degree=6,
)

xi_qp, W_q = get_quadrature(fe_type=fe_type)
phi_qn, dphi_dxi_qnp = eval_basis_and_derivatives(fe_type=fe_type, xi_qp=xi_qp)
Q = get_quadrature(fe_type=fe_type)[0].shape[0] 


V=points.shape[0]
U=3 # num of solution componets
x_end = mesh_to_jax(vertices=points, cells=cells)
#assembly_map_b = mesh_to_sparse_assembly_map(n_vertices=V, cells=cells)
E = x_end.shape[0] # num elements

# Material properties 

cell_domain_ids=mesh.cell_data["gmsh:physical"][0]-1
#material_param=jnp.array([(4.76e3,4.76e3,4.76e3,1.737e3,1.737e3,1.737e3,0.37,0.37,0.37),
 #                         (276e3, 19.5e3, 19.5e3, 70e3, 70e3, 5.735e3, 0.28, 0.28, 0.70)])

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
        
        R_Sig = jnp.array([
            [c**2, s**2, 0, 0, 0, -2*cs],
            [s**2, c**2, 0, 0, 0, 2*cs],
            [0,    0,    1, 0, 0, 0    ],
            [0,    0,    0, c, s, 0    ],
            [0,    0,    0, -s, c, 0   ],
            [cs,  -cs,   0, 0, 0, c**2 - s**2]
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
    
@partial(jax.jit, static_argnames=['num_quad_points'])
def get_heterogeneous_C_matrix(cell_domain_ids, num_quad_points, material_param, domain_angles):

    C_matrices_base = jax.vmap(build_single_C_matrix)(material_param)
    
    C_matrices_rotated = jax.vmap(rotate_C_matrix)(C_matrices_base, domain_angles)
    
    return C_matrices_rotated[cell_domain_ids]


C_ess = get_heterogeneous_C_matrix(
    cell_domain_ids, 
    num_quad_points=Q, 
    material_param=material_param, 
    domain_angles=angles
)

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
    periodic_assembly_map: object, # Use the periodic map!
    unique_dofs_full: jnp.ndarray, # Pass your unique indices (including Lagrange)
    u_f: jnp.ndarray,   
):
    u_g_flat, lamb_vec = u_f[:-3], u_f[-3:]
    
    E=x_end.shape[0]
    u_end = transform_global_unraveled_to_element_node(periodic_assembly_map, u_g_flat, E)
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
            assembly_map=periodic_assembly_map, 
            v_en=R_one_case_EN3
        )
    R_global_cases = jax.vmap(assemble_one_case)(R_cases_first)
    R_elastic_matrix = R_global_cases.reshape(6, -1).T
    constraint_rhs = jnp.zeros((3, 6))
    R_mixed_matrix = jnp.concatenate([R_elastic_matrix, constraint_rhs], axis=0)
    return R_mixed_matrix[unique_dofs_full]

# Periodic assembly map
periodic_map_b, dof_map_np = mesh_to_periodic_sparse_assembly_map(V, cells, points,tol=1e-6)
unique_dofs = jnp.unique(dof_map_np)
n_total_dofs = len(dof_map_np)


u_0_g_full = jnp.zeros(shape=(V * U + 3)) 
R_f_all_cases = lambda u_f: calculate_residual_batch_element_kernel_mixed_periodic(
    x_end,
    dphi_dxi_qnp,
    phi_qn,
    W_q,
    C_ess,
    periodic_map_b, # Pass your updated map here
    unique_dofs,   # Pass the slice indices
    u_f=u_f
)

R_f_reduced = R_f_all_cases(u_f=u_0_g_full)

@jax.jit
def _calculate_jacobian_batch_element_kernel_periodic(
    x_end: jnp.ndarray,
    u_mixed_f,
    dphi_dxi_qnp: jnp.ndarray,
    phi_qn: jnp.ndarray,          
    W_q: jnp.ndarray,
    C_ess: jnp.ndarray,
    periodic_assembly_map: object,
):
    E = x_end.shape[0]
    u_f = u_mixed_f[:-3]
    lamb_f = u_mixed_f[-3:]
    u_enu = transform_global_unraveled_to_element_node(periodic_assembly_map, u_f, E)

    N = x_end.shape[1]
    D = x_end.shape[2]
    U = u_enu.shape[2]
    u_et = u_enu.reshape(E, N * U)
    lamb_et = jnp.tile(lamb_f, (E, 1)) 
    u_mixed_et = jnp.concatenate([u_et, lamb_et], axis=1) # Shape: (E, 12)
    
    @jax.jit
    def residual_kernel(u_mixed_local, x_nd, C_ss):
        u_t = u_mixed_local[:N * U]
        l_t = u_mixed_local[N * U:] # These are the 3 Lagrange multipliers
        u_nd = u_t.reshape(N, U)

        R_elastic, f_vol_element = _element_residual_single_case(
        u_nd,
        x_nd,
        dphi_dxi_qnp,
        phi_qn,
        W_q,
        C_ss,
        epsilon_bar= jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        R_u_local = R_elastic + (f_vol_element * l_t[None, :])
        R_lambda_local = jnp.sum(u_nd * f_vol_element, axis=0)
        
        return jnp.concatenate([R_u_local.ravel(), R_lambda_local.ravel()])

    J_mixed_elements = jax.vmap(jax.jacfwd(residual_kernel, argnums=0))(
        u_mixed_et, x_end, C_ess
    )
    
    return J_mixed_elements

u_0_g=jnp.zeros(shape=(V*U+3)) 

J_ett =_calculate_jacobian_batch_element_kernel_periodic(
    x_end,
    u_0_g,
    dphi_dxi_qnp,
    phi_qn,          
    W_q,
    C_ess,
    periodic_map_b,
    )


def ebe_jacobian_product_periodic(J_mixed_elements, periodic_assembly_map: object, unique_dofs, n_total_dofs,  z_reduced):
    z_full = jnp.zeros(n_total_dofs).at[unique_dofs].set(z_reduced, unique_indices=True)

    # 1. SPLIT z_global
    z_u = z_full[:-3]    # Displacement part
    z_lamb = z_full[-3:] # Lagrange multiplier part

    z_u_enu = transform_global_unraveled_to_element_node(
        periodic_assembly_map, z_u, J_mixed_elements.shape[0]
    )

    N,U=  z_u_enu.shape[1], z_u_enu.shape[2]
    z_u_local = z_u_enu.reshape(-1, N*U)
    z_lamb_et = jnp.tile(z_lamb, (z_u_local.shape[0], 1))
    z_mixed_local = jnp.concatenate([z_u_local, z_lamb_et], axis=1)

    # 4. MULTIPLY: (E, 12, 12) @ (E, 12) -> (E, 12)
    f_mixed_local = jnp.einsum('eij,ej->ei', J_mixed_elements, z_mixed_local)

    # 5. SCATTER: Local forces to Global vector
    # Separate the results: f_u is (E, 9), f_lamb is (E, 3)
    f_u_local = f_mixed_local[:, :N*U].reshape(-1, N, U) # Reshape back for your sum function
    f_lamb_local = f_mixed_local[:, N*U:]

    # Use your existing sum function for the displacement part
    f_u_global_full = transform_element_node_to_global_unraveled_sum(
        assembly_map=periodic_assembly_map, v_en=f_u_local
    )

    # Sum the element-level Lagrange contributions into the final 3 slots
    f_lamb_global = jnp.sum(f_lamb_local, axis=0)
    f_full = jnp.concatenate([f_u_global_full, f_lamb_global])
    return f_full[unique_dofs]


periodic_operator = jax.tree_util.Partial(
    ebe_jacobian_product_periodic, 
    J_ett, 
    periodic_map_b,
    unique_dofs,
    n_total_dofs,
)
def solve_single_case(b_col):
    # A = ebe_operator provides the J*z logic for bicgstab
    delta_u, info = jax.scipy.sparse.linalg.bicgstab(
        A=periodic_operator,  
        b=b_col,
        tol=1e-10,
        atol=1e-10,
        maxiter=1000
    )
    return delta_u
    
batch_solver = jax.vmap(solve_single_case, in_axes=1, out_axes=1)
delta_u_matrix = batch_solver(-R_f_reduced)


# 4. Compute D1
D1 = jnp.einsum('ni,nj->ij', delta_u_matrix, R_f_reduced)

@jax.jit
def compute_global_D_bar(
    x_end: jnp.ndarray,        # (E, N, D) - Element coordinates
    dphi_dxi_qnp: jnp.ndarray, # (Q, N, D) - Ref shape derivatives
    W_q: jnp.ndarray,          # (Q,) - Quadrature weights
    C_ess: jnp.ndarray         # (E, 6, 6) - The pre-computed rotated stiffness matrices!
):
    def _element_D_bar(x_nd, C_ss):
        # 1. Compute Jacobian and element volume
        J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
        det_J_q = jnp.linalg.det(J_qdp)
        elem_vol = jnp.sum(det_J_q * W_q)
        D_bar_element = C_ss * elem_vol
        
        return D_bar_element, elem_vol

    batch_D_bar_fn = jax.vmap(_element_D_bar, in_axes=(0, 0))
    D_bar_elements, elem_vols = batch_D_bar_fn(x_end, C_ess)
    
    return jnp.sum(D_bar_elements, axis=0), jnp.sum(elem_vols) 

D_bar, omega = compute_global_D_bar(x_end, dphi_dxi_qnp, W_q, C_ess)
D_eff= (D_bar+D1)/omega # Effective Stiffness Matrix

I = jnp.eye(6, dtype=D_eff.dtype)
Com = jnp.linalg.solve(D_eff, I)

E1,E2,E3=1/Com[0,0], 1/Com[1,1], 1/Com[2,2]
v12,v13,v23=-Com[0,1]/Com[0,0], -Com[0,2]/Com[0,0], -Com[1,2]/Com[1,1]
G23,G13,G12=1/Com[3,3], 1/Com[4,4], 1/Com[5,5]
print('Time taken', time.time()- start, '\n')
props=[E1,E2,E3,G12,G13,G23,v12,v13,v23];

labels = ["E1", "E2", "E3", "G12", "G13", "G23", "v12", "v13", "v23"];
print("--- Effective Material Properties ---")
for label, val in zip(labels, props):
    print(f"{label}: {val}")
print('\n Effective Stiffness matrix \n')
np.set_printoptions(precision=6)
print(D_eff)
