import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.linalg import solve

# --- 1. Geometry and Properties ---
nodes_3d = np.array([
    [0.0, 0.0, 0.0],  # Node 0
    [1.0, 0.0, 0.0],  # Node 1
    [2.0, 0.0, 0.0]   # Node 2
])

elements_3d = np.array([
    [0, 1],
    [1, 2]
])

outer_radius = 0.1
inner_radius = 0.08

material_properties_3d = {
    "youngs_modulus": 200e9,
    "poissons_ratio": 0.27
}

A = np.pi * (outer_radius**2 - inner_radius**2)
Iy = np.pi / 4 * (outer_radius**4 - inner_radius**4)
Iz = Iy
J = 2 * Iy
cross_section_properties_3d = {
    "Area": A, "Iy": Iy, "Iz": Iz, "J": J,
    "outer_radius": outer_radius, "inner_radius": inner_radius
}

# --- 2. Element Stiffness Matrix Function ---
def calculate_element_stiffness_matrix_3d_beam(node1, node2, mat, cs):
    E = mat['youngs_modulus']
    G = E / (2 * (1 + mat['poissons_ratio']))
    A, Iy, Iz, J = cs['Area'], cs['Iy'], cs['Iz'], cs['J']
    L = np.linalg.norm(node2 - node1)
    if L < 1e-9: return np.zeros((12, 12))

    Ke_local = np.zeros((12, 12))
    # Axial
    axial = E * A / L
    Ke_local[0,0] = Ke_local[6,6] = axial; Ke_local[0,6] = Ke_local[6,0] = -axial
    # Torsion
    torsion = G * J / L
    Ke_local[3,3] = Ke_local[9,9] = torsion; Ke_local[3,9] = Ke_local[9,3] = -torsion
    # Bending Z
    bz1, bz2, bz3, bz4 = 12*E*Iz/L**3, 6*E*Iz/L**2, 4*E*Iz/L, 2*E*Iz/L
    Ke_local[1,1] = Ke_local[7,7] = bz1; Ke_local[1,7] = Ke_local[7,1] = -bz1
    Ke_local[1,5] = Ke_local[5,1] = Ke_local[1,11] = Ke_local[11,1] = bz2
    Ke_local[5,5] = Ke_local[11,11] = bz3; Ke_local[5,11] = Ke_local[11,5] = bz4
    Ke_local[5,7] = Ke_local[7,5] = Ke_local[7,11] = Ke_local[11,7] = -bz2
    # Bending Y
    by1, by2, by3, by4 = 12*E*Iy/L**3, 6*E*Iy/L**2, 4*E*Iy/L, 2*E*Iy/L
    Ke_local[2,2] = Ke_local[8,8] = by1; Ke_local[2,8] = Ke_local[8,2] = -by1
    Ke_local[2,4] = Ke_local[4,2] = Ke_local[2,10] = Ke_local[10,2] = -by2
    Ke_local[4,4] = Ke_local[10,10] = by3; Ke_local[4,10] = Ke_local[10,4] = by4
    Ke_local[4,8] = Ke_local[8,4] = Ke_local[8,10] = Ke_local[10,8] = by2
    return Ke_local

# --- 3. Stress and Force Calculation ---
def calculate_element_results_3d_beam(node1_coords_global, node2_coords_global, element_global_indices, nodal_displacements, material_properties, cross_section_properties):
    E = material_properties['youngs_modulus']
    nu = material_properties['poissons_ratio']
    G = E / (2 * (1 + nu))

    A = cross_section_properties['Area']
    Iy = cross_section_properties['Iy']
    Iz = cross_section_properties['Iz']
    J = cross_section_properties['J']
    outer_radius = cross_section_properties.get('outer_radius')
    inner_radius = cross_section_properties.get('inner_radius', 0.0)

    vector = np.array(node2_coords_global) - np.array(node1_coords_global)
    L = np.linalg.norm(vector)

    if L < 1e-9:
        return {
            'local_forces_moments': np.zeros(12),
            'stress_components': {},
            'von_mises_stresses': {}
        }

    # Define orthonormal local coordinate axes
    if np.isclose(np.abs(vector[0]), L):  # Beam along global x-axis
        local_x_axis = np.array([1.0 if vector[0] > 0 else -1.0, 0.0, 0.0])
        local_y_axis = np.array([0.0, 1.0, 0.0])
        local_z_axis = np.array([0.0, 0.0, 1.0])
    elif np.isclose(np.abs(vector[1]), L):  # Beam along global y-axis
        local_x_axis = np.array([0.0, 1.0 if vector[1] > 0 else -1.0, 0.0])
        local_y_axis = np.array([1.0, 0.0, 0.0])
        local_z_axis = np.array([0.0, 0.0, 1.0])
    elif np.isclose(np.abs(vector[2]), L):  # Beam along global z-axis
        local_x_axis = np.array([0.0, 0.0, 1.0 if vector[2] > 0 else -1.0])
        local_y_axis = np.array([1.0, 0.0, 0.0])
        local_z_axis = np.array([0.0, 1.0, 0.0])
    else:  # General orientation
        local_x_axis = vector / L
        global_z = np.array([0.0, 0.0, 1.0])
        local_y_axis = np.cross(global_z, local_x_axis)
        if np.linalg.norm(local_y_axis) < 1e-9:
            global_y = np.array([0.0, 1.0, 0.0])
            local_y_axis = np.cross(global_y, local_x_axis)
        local_y_axis = local_y_axis / np.linalg.norm(local_y_axis)
        local_z_axis = np.cross(local_x_axis, local_y_axis)
        local_z_axis = local_z_axis / np.linalg.norm(local_z_axis)

    R = np.array([local_x_axis, local_y_axis, local_z_axis])  # Rotation matrix from global to local

    # Extract element global displacements (12 DOFs)
    dofs_per_node_global = 6
    element_global_dofs = []
    for node_global_idx in element_global_indices:
        element_global_dofs.extend([
            node_global_idx * dofs_per_node_global + i for i in range(6)
        ])

    element_global_displacements = nodal_displacements[element_global_dofs]

    # Transform displacements to local coordinate system
    T_node_6x6 = np.zeros((6, 6))
    T_node_6x6[:3, :3] = R
    T_node_6x6[3:, 3:] = R

    T = np.zeros((12, 12))
    T[:6, :6] = T_node_6x6
    T[6:12, 6:12] = T_node_6x6

    element_local_displacements = T @ element_global_displacements

    # Local stiffness matrix calculation
    Ke_local = np.zeros((12, 12))
    axial_stiffness = E * A / L
    Ke_local[0, 0] = Ke_local[6, 6] = axial_stiffness
    Ke_local[0, 6] = Ke_local[6, 0] = -axial_stiffness

    torsional_stiffness = G * J / L
    Ke_local[3, 3] = Ke_local[9, 9] = torsional_stiffness
    Ke_local[3, 9] = Ke_local[9, 3] = -torsional_stiffness

    bending_stiffness_z1 = 12 * E * Iz / L**3
    bending_stiffness_z2 = 6 * E * Iz / L**2
    bending_stiffness_z3 = 4 * E * Iz / L
    bending_stiffness_z4 = 2 * E * Iz / L

    Ke_local[1, 1] = Ke_local[7, 7] = bending_stiffness_z1
    Ke_local[1, 5] = Ke_local[5, 1] = bending_stiffness_z2
    Ke_local[1, 7] = Ke_local[7, 1] = -bending_stiffness_z1
    Ke_local[1, 11] = Ke_local[11, 1] = bending_stiffness_z2
    Ke_local[5, 5] = bending_stiffness_z3
    Ke_local[5, 7] = Ke_local[7, 5] = -bending_stiffness_z2
    Ke_local[5, 11] = Ke_local[11, 5] = bending_stiffness_z4
    Ke_local[7, 11] = Ke_local[11, 7] = -bending_stiffness_z2
    Ke_local[11, 11] = bending_stiffness_z3

    bending_stiffness_y1 = 12 * E * Iy / L**3
    bending_stiffness_y2 = 6 * E * Iy / L**2
    bending_stiffness_y3 = 4 * E * Iy / L
    bending_stiffness_y4 = 2 * E * Iy / L

    Ke_local[2, 2] = Ke_local[8, 8] = bending_stiffness_y1
    Ke_local[2, 4] = Ke_local[4, 2] = -bending_stiffness_y2
    Ke_local[2, 8] = Ke_local[8, 2] = -bending_stiffness_y1
    Ke_local[2, 10] = Ke_local[10, 2] = -bending_stiffness_y2
    Ke_local[4, 4] = bending_stiffness_y3
    Ke_local[4, 8] = Ke_local[8, 4] = bending_stiffness_y2
    Ke_local[4, 10] = Ke_local[10, 4] = bending_stiffness_y4
    Ke_local[8, 10] = Ke_local[10, 8] = bending_stiffness_y2
    Ke_local[10, 10] = bending_stiffness_y3

    # Calculate internal forces
    element_local_forces_moments = Ke_local @ element_local_displacements

    # Evaluate stresses at representative extreme points on the cross-section
    stress_components = {}
    von_mises_stresses = {}

    if outer_radius is None:
        return {
            'local_forces_moments': element_local_forces_moments,
            'stress_components': {},
            'von_mises_stresses': {}
        }

    points_on_cross_section = [
        {'description': 'y_plus', 'coords_local': [outer_radius, 0.0]},
        {'description': 'y_minus', 'coords_local': [-outer_radius, 0.0]},
        {'description': 'z_plus', 'coords_local': [0.0, outer_radius]},
        {'description': 'z_minus', 'coords_local': [0.0, -outer_radius]}
    ]

    # Calculate stresses at Node 2 end
    Fx2 = element_local_forces_moments[6]
    Mx2 = element_local_forces_moments[9]
    My2 = element_local_forces_moments[10]
    Mz2 = element_local_forces_moments[11]

    for point in points_on_cross_section:
        desc = point['description']
        y_local, z_local = point['coords_local']

        # Normal stress: Axial + Bending Y + Bending Z
        sigma_x_axial = Fx2 / A if A != 0 else 0
        sigma_x_bending_z = (-Mz2 * y_local) / Iz if Iz != 0 else 0
        sigma_x_bending_y = (My2 * z_local) / Iy if Iy != 0 else 0
        sigma_x_prime = sigma_x_axial + sigma_x_bending_z + sigma_x_bending_y

        # Shear stress from torsion
        r = np.sqrt(y_local**2 + z_local**2)
        tau_torsion_magnitude = (np.abs(Mx2) * r) / J if J != 0 and r != 0 else 0

        # Torsional shear components
        tau_x_prime_y_prime = -tau_torsion_magnitude * (z_local / r) if r != 0 else 0
        tau_x_prime_z_prime = tau_torsion_magnitude * (y_local / r) if r != 0 else 0

        # Von Mises: sqrt(sigma^2 + 3 * tau^2)
        von_mises = np.sqrt(sigma_x_prime**2 + 3 * tau_torsion_magnitude**2)

        stress_components[desc] = (sigma_x_prime, 0.0, 0.0, tau_x_prime_y_prime, 0.0, tau_x_prime_z_prime)
        von_mises_stresses[desc] = von_mises

    return {
        'local_forces_moments': element_local_forces_moments,
        'stress_components': stress_components,
        'von_mises_stresses': von_mises_stresses
    }

# --- 4. Assembly and Solve ---
num_nodes = len(nodes_3d)
K = np.zeros((num_nodes*6, num_nodes*6))
F = np.zeros(num_nodes*6)
F[2*6 + 1] = -1000.0  # Downward load of 1000 N at Node 2 in Y-direction

for idx, (n1, n2) in enumerate(elements_3d):
    # Element stiffness in global coordinate system (beam is oriented along global X, so Ke_global = Ke_local)
    Ke = calculate_element_stiffness_matrix_3d_beam(nodes_3d[n1], nodes_3d[n2], material_properties_3d, cross_section_properties_3d)
    dofs = np.concatenate([np.arange(n1*6, n1*6+6), np.arange(n2*6, n2*6+6)])
    for i in range(12):
        for j in range(12):
            K[dofs[i], dofs[j]] += Ke[i, j]

# Boundary Conditions: Fully fix Node 0
fixed_dofs = np.arange(0, 6)
for dof in fixed_dofs:
    K[dof, :] = 0; K[:, dof] = 0; K[dof, dof] = 1; F[dof] = 0

# Solve linear system
U = solve(K, F)
print("System Solved Successfully.\")")

# Calculate stress results for elements
element_results = {}
for element_idx, element_global_indices in enumerate(elements_3d):
    node1_coords = nodes_3d[element_global_indices[0]]
    node2_coords = nodes_3d[element_global_indices[1]]
    res = calculate_element_results_3d_beam(
        node1_coords, node2_coords, element_global_indices,
        U, material_properties_3d, cross_section_properties_3d
    )
    element_results[element_idx] = res

# Display stress summaries
print("\nElement Stress Analysis (Von Mises Equivalent Stress):")
for element_idx, res in element_results.items():
    if res['von_mises_stresses']:
        max_vm = max(res['von_mises_stresses'].values())
        print(f"Element {element_idx} (Nodes {elements_3d[element_idx]}): Max Von Mises Stress = {max_vm:.3f} Pa ({max_vm / 1e6:.3f} MPa)")
        for desc, val in res['von_mises_stresses'].items():
            print(f"  - Position '{desc}': {val / 1e6:.3f} MPa")

# --- 5. 3D Color-Coded Tube Visualization ---
def visualize_3d_beam_results(nodes_3d, elements_3d, nodal_displacements, element_results_dict, result_type='von_mises', scale_factor=50.0, cross_section_properties=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'3D FEA Results: {result_type.replace("_", " ").title()} stress mapping')
    
    # Orthonormalizing display axes
    max_range = np.array([
        nodes_3d[:,0].max() - nodes_3d[:,0].min(),
        nodes_3d[:,1].max() - nodes_3d[:,1].min(),
        nodes_3d[:,2].max() - nodes_3d[:,2].min()
    ]).max()
    mid_x = (nodes_3d[:,0].max() + nodes_3d[:,0].min()) * 0.5
    mid_y = (nodes_3d[:,1].max() + nodes_3d[:,1].min()) * 0.5
    mid_z = (nodes_3d[:,2].max() + nodes_3d[:,2].min()) * 0.5
    plot_range = max(max_range * 0.6, 0.5)
    ax.set_xlim(mid_x - plot_range, mid_x + plot_range)
    ax.set_ylim(mid_y - plot_range, mid_y + plot_range)
    ax.set_zlim(mid_z - plot_range, mid_z + plot_range)

    # Calculate deformed coordinates
    displacements_reshaped = nodal_displacements.reshape(-1, 6)
    deformed_nodes = nodes_3d + displacements_reshaped[:, :3] * scale_factor

    # Fetch result values for colormap
    all_result_values = []
    elements_to_visualize = []
    
    for element_idx, results in element_results_dict.items():
        if results and results['von_mises_stresses']:
            result_value = max(results['von_mises_stresses'].values())
            all_result_values.append(result_value)
            elements_to_visualize.append(element_idx)

    all_result_values = np.array(all_result_values)
    
    # Norm definition for gradient coloring (Classic Jet Colormap for structural analysis)
    if all_result_values.max() - all_result_values.min() < 1e-9:
        norm = plt.Normalize(all_result_values.min() - 1.0, all_result_values.max() + 1.0)
    else:
        norm = plt.Normalize(all_result_values.min(), all_result_values.max())
    
    cmap = plt.cm.jet  # Classic structural gradient scale

    resolution = 16  # Mesh density for cylindrical tube representation
    outer_rad = cross_section_properties.get('outer_radius', 0.1) if cross_section_properties else 0.1
    inner_rad = cross_section_properties.get('inner_radius', 0.0) if cross_section_properties else 0.0

    # Draw elements as full color-coded tubes
    for i, element_idx in enumerate(elements_to_visualize):
        node_indices = elements_3d[element_idx]
        pt1 = deformed_nodes[node_indices[0]]
        pt2 = deformed_nodes[node_indices[1]]
        
        color = cmap(norm(all_result_values[i]))
        vector = pt2 - pt1
        mag = np.linalg.norm(vector)
        if mag < 1e-9: continue
        
        vector_norm = vector / mag
        # Build local coordinate transformation frame for generating cylinder mesh points
        if np.isclose(np.abs(vector_norm[2]), 1.0):
            v1 = np.array([1.0, 0.0, 0.0])
        else:
            v1 = np.array([0.0, 0.0, 1.0])
            
        v1 = v1 - np.dot(v1, vector_norm) * vector_norm
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(vector_norm, v1)

        theta = np.linspace(0, 2 * np.pi, resolution)
        circle_points_outer = outer_rad * (np.outer(np.cos(theta), v1) + np.outer(np.sin(theta), v2))
        circle_points_inner = inner_rad * (np.outer(np.cos(theta), v1) + np.outer(np.sin(theta), v2))

        # Generate outer cylindrical wall surface patches
        verts_outer = []
        for j in range(resolution - 1):
            verts_outer.append([
                pt1 + circle_points_outer[j, :],
                pt1 + circle_points_outer[j+1, :],
                pt2 + circle_points_outer[j+1, :],
                pt2 + circle_points_outer[j, :]
            ])
        verts_outer.append([
            pt1 + circle_points_outer[resolution-1, :],
            pt1 + circle_points_outer[0, :],
            pt2 + circle_points_outer[0, :],
            pt2 + circle_points_outer[resolution-1, :]
        ])

        # Plot outer surface
        ax.add_collection3d(Poly3DCollection(verts_outer, facecolor=color, edgecolor='k', linewidths=0.2, alpha=0.9))

        # Generate inner wall hollow patches if it's a hollow tube
        if inner_rad > 0:
            verts_inner = []
            for j in range(resolution - 1):
                verts_inner.append([
                    pt1 + circle_points_inner[j, :],
                    pt1 + circle_points_inner[j+1, :],
                    pt2 + circle_points_inner[j+1, :],
                    pt2 + circle_points_inner[j, :]
                ])
            verts_inner.append([
                pt1 + circle_points_inner[resolution-1, :],
                pt1 + circle_points_inner[0, :],
                pt2 + circle_points_inner[0, :],
                pt2 + circle_points_inner[resolution-1, :]
            ])
            ax.add_collection3d(Poly3DCollection(verts_inner, facecolor='gray', edgecolor='k', linewidths=0.1, alpha=0.5))

    # Add interactive reference nodes
    ax.scatter(deformed_nodes[:, 0], deformed_nodes[:, 1], deformed_nodes[:, 2], color='black', s=100, zorder=5, label='Nodal Positions')
    for idx, pt in enumerate(deformed_nodes):
        ax.text(pt[0] + 0.05, pt[1] + 0.05, pt[2] + 0.05, f"N{idx}", color='black', fontweight='bold')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    
    # Render colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(all_result_values)
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Von Mises Equivalent Stress (Pa)')
    
    plt.grid(True)
    plt.savefig('fea_plot.png', dpi=300, bbox_inches='tight')
    print("\nColor-coded structural analysis plot successfully saved as 'fea_plot.png'.")
    plt.show()

# Run the stress visualizer
visualize_3d_beam_results(
    nodes_3d, elements_3d, U, element_results,
    result_type='von_mises', scale_factor=50.0,
    cross_section_properties=cross_section_properties_3d
)
