import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.linalg import solve
import pandas as pd

# Set premium dark-themed page config
st.set_page_config(
    page_title="3D spatial Structural FEA Engine",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling rules
st.markdown("""
<style>
    .reportview-container {
        background: #0b0f19;
    }
    .metric-card {
        background: #1e2942;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
    }
    .metric-val {
        font-size: 2.25rem;
        font-weight: 700;
        color: #06b6d4;
        margin-top: 0.5rem;
    }
    .metric-title {
        color: #9ca3af;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ 3D Beam Finite Element Analysis (FEA) Engine")
st.markdown("A premium, real-time spatial simulator to calculate nodal displacements, coordinate system transformations, stress distributions, and volumetric yielding limits under combined loading cases.")

# --- SIDEBAR: Documentation ---
@st.dialog("📖 Technical Documentation", width="large")
def show_documentation():
    st.markdown(r"""
### Introduction
This program implements an advanced, highly modular **3D Finite Element Analysis (FEA) Solver** specifically designed for representing, simulating, and evaluating spatial structural frames using 3D linear beam elements.

Equipped with support for elastic material bounds and customized cross-sectional profile calculations (circular hollow tube structure), the engine is capable of computing structural nodal displacements, internal forces, torsional shearing stresses, and yielding metrics under arbitrary point loading forces.

**6 Degrees of Freedom**
Evaluates translation and rotational displacements simultaneously for every node in local and global spatial axes.

**Structural Hotspotting**
Post-processes combined loading stress tensors to calculate Von Mises equivalent stress mapped to 3D tube geometries.

---

### Mathematical Formulations
The mathematical pipeline utilizes Euler-Bernoulli linear bending mechanics combined with St. Venant torsional formulations. Transverse shear and axial tension stiffness properties are aggregated into a $12 \times 12$ element stiffness matrix $K^e_{local}$ in the local coordinate systems.

**Governing Nodal Equilibrium Equation**
$$f_{local} = K^e_{local} \cdot u_{local}$$

To transition elements from arbitrary orientations in 3D space to the global system, direction cosines are used to construct an orthonormal coordinate rotation matrix $R_{3\times 3}$. The global element stiffness matrix is derived via: 

**Orthogonal Coordinate Transformation**
$$K^e_{global} = T^T \cdot K^e_{local} \cdot T$$

Where $T$ is the $12\times 12$ kinematic transformation matrix composed of the direction cosines rotation matrix block-diagonally.

#### Von Mises Equivalent Stress
During the post-processing phase, the stress state at extreme fibers is calculated by combining axial force $F_x$, torsion $M_x$, and bending moments $M_y$, $M_z$. The normal stress $\sigma_x$ and shearing stress $\tau_{torsion}$ are computed as:

**Combined Axial and Biaxial Bending Normal Stress**
$$\sigma_x = \frac{F_x}{A} - \frac{M_z \cdot y}{I_z} + \frac{M_y \cdot z}{I_y}$$

**Torsional Shearing Stress Formula**
$$\tau_{torsion} = \frac{M_x \cdot r_{outer}}{J}$$

Finally, these are reduced into the uni-axial equivalent **Von Mises Stress** at the critical cross-sectional outer points:

**Von Mises Equivalent Yield Condition**
$$\sigma_{VM} = \sqrt{\sigma_x^2 + 3 \tau_{torsion}^2}$$

---

### Code Architecture
The code is designed to be lean, performant, and self-contained within `main.py`. The structural execution workflow is separated into 5 clear pipelines:

**1. Pre-Processing & Geometry Setup**
Defining nodes, connectivity arrays, hollow tube radius bounds, and isotropic material variables (Elastic modulus & Poisson ratio).

**2. Local Element Stiffness Formulations**
The function `calculate_element_stiffness_matrix_3d_beam` evaluates the structural resistances based on length $L$, cross-section area $A$, moments of inertia $I_y, I_z$, and polar moment of inertia $J$.

**3. Assembly of Global System Equations**
Degrees of Freedom (DOFs) are mapped to global indices, and the element matrices are summed into the global stiffness matrix $K_{global}$ ($N_{nodes} \times 6$).

**4. Boundary Enforcement and Solving**
Fixed supports are applied via penalty or partition boundaries (zeroing rows/columns and setting diagonal elements to 1). The displacement vector $U$ is solved via `scipy.linalg.solve`.

**5. Stress Post-Processing and 3D Volumetric Tube Visualization**
Nodal translations are mapped to elements. Element cylinder vertices are created in 3D, colored based on Von Mises output, and plotted utilizing matplotlib's `Poly3DCollection`.
""")

if st.sidebar.button("📖 View Technical Documentation", use_container_width=True):
    show_documentation()

# --- SIDEBAR: Configuration Panel ---
st.sidebar.header("⚙️ Design Configuration")

# Material presets
st.sidebar.subheader("💎 Material Selection")
material_preset = st.sidebar.selectbox(
    "Choose Material Preset",
    ["Structural Steel", "Aluminum 6061-T6", "Titanium Grade 5", "Custom Isotropic"]
)

# Setup material constants based on selection
if material_preset == "Structural Steel":
    youngs_modulus = 200e9
    poissons_ratio = 0.27
    yield_strength = 250e6  # Pa
elif material_preset == "Aluminum 6061-T6":
    youngs_modulus = 68.9e9
    poissons_ratio = 0.33
    yield_strength = 276e6  # Pa
elif material_preset == "Titanium Grade 5":
    youngs_modulus = 114e9
    poissons_ratio = 0.34
    yield_strength = 880e6  # Pa
else:
    youngs_modulus = st.sidebar.number_input("Young's Modulus (E) in GPa", min_value=1.0, max_value=1000.0, value=200.0) * 1e9
    poissons_ratio = st.sidebar.slider("Poisson's Ratio (ν)", 0.0, 0.49, 0.27)
    yield_strength = st.sidebar.number_input("Yield Strength in MPa", min_value=10.0, max_value=5000.0, value=250.0) * 1e6

# Geometry Configuration
st.sidebar.subheader("📐 Beam Geometry (Hollow Tube)")
outer_radius = st.sidebar.slider("Outer Radius (m)", 0.02, 0.50, 0.10, step=0.01)
inner_radius = st.sidebar.slider("Inner Radius (m)", 0.01, outer_radius - 0.01, 0.08, step=0.01)
beam_length = st.sidebar.slider("Total Beam Length (m)", 0.5, 10.0, 2.0, step=0.1)
num_elements = st.sidebar.slider("Number of Finite Elements", 1, 10, 2, step=1)

# Combined Loadings Configuration
st.sidebar.subheader("⚡ Tip Load Conditions (Node Tip)")
fx_load = st.sidebar.number_input("Axial Force Fx (N) [+ Tension]", value=0.0)
fy_load = st.sidebar.number_input("Transverse Shear Force Fy (N) [Vertical]", value=-1000.0)
fz_load = st.sidebar.number_input("Transverse Shear Force Fz (N) [Lateral]", value=0.0)
mx_load = st.sidebar.number_input("Torsional Moment Mx (N-m)", value=0.0)

# Vis Settings
st.sidebar.subheader("🎨 Visual Scaling")
scale_factor = st.sidebar.slider("Deflection Display Scale Factor (x)", 1.0, 500.0, 50.0)

# --- 1. Math Precomputation ---
A = np.pi * (outer_radius**2 - inner_radius**2)
Iy = np.pi / 4 * (outer_radius**4 - inner_radius**4)
Iz = Iy
J = 2 * Iy

cross_section_properties = {
    "Area": A, "Iy": Iy, "Iz": Iz, "J": J,
    "outer_radius": outer_radius, "inner_radius": inner_radius
}

material_properties = {
    "youngs_modulus": youngs_modulus,
    "poissons_ratio": poissons_ratio
}

# --- 2. Build 3D Mesh ---
x_coords = np.linspace(0.0, beam_length, num_elements + 1)
nodes_3d = np.zeros((num_elements + 1, 3))
nodes_3d[:, 0] = x_coords  # Straight horizontal along X-axis

elements_3d = []
for i in range(num_elements):
    elements_3d.append([i, i + 1])
elements_3d = np.array(elements_3d)

# --- 3. Stiffness Matrix Calculator ---
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

# --- 4. Global System Assembly & Solution ---
num_nodes = len(nodes_3d)
K = np.zeros((num_nodes*6, num_nodes*6))
F = np.zeros(num_nodes*6)

# Apply forces at the very tip (last node)
tip_node_idx = num_nodes - 1
F[tip_node_idx*6 + 0] = fx_load
F[tip_node_idx*6 + 1] = fy_load
F[tip_node_idx*6 + 2] = fz_load
F[tip_node_idx*6 + 3] = mx_load

for idx, (n1, n2) in enumerate(elements_3d):
    Ke = calculate_element_stiffness_matrix_3d_beam(nodes_3d[n1], nodes_3d[n2], material_properties, cross_section_properties)
    dofs = np.concatenate([np.arange(n1*6, n1*6+6), np.arange(n2*6, n2*6+6)])
    for i in range(12):
        for j in range(12):
            K[dofs[i], dofs[j]] += Ke[i, j]

# Fix boundary condition (Fully fixed Node 0 support)
fixed_dofs = np.arange(0, 6)
for dof in fixed_dofs:
    K[dof, :] = 0; K[:, dof] = 0; K[dof, dof] = 1; F[dof] = 0

# Solver execution
U = solve(K, F)

# --- 5. Stress Post-Processing ---
def calculate_element_results_3d_beam(node1_coords_global, node2_coords_global, element_global_indices, nodal_displacements, mat, cs):
    E = mat['youngs_modulus']
    nu = mat['poissons_ratio']
    G = E / (2 * (1 + nu))
    A, Iy, Iz, J = cs['Area'], cs['Iy'], cs['Iz'], cs['J']
    ro = cs['outer_radius']
    ri = cs['inner_radius']

    vector = np.array(node2_coords_global) - np.array(node1_coords_global)
    L = np.linalg.norm(vector)
    if L < 1e-9:
        return {'local_forces_moments': np.zeros(12), 'von_mises_stresses': {'max': 0.0}}

    # Coordinate system direction cosines rotation
    if np.isclose(np.abs(vector[0]), L):
        local_x = np.array([1.0 if vector[0] > 0 else -1.0, 0.0, 0.0])
        local_y = np.array([0.0, 1.0, 0.0])
        local_z = np.array([0.0, 0.0, 1.0])
    else:
        local_x = vector / L
        global_z = np.array([0.0, 0.0, 1.0])
        local_y = np.cross(global_z, local_x)
        if np.linalg.norm(local_y) < 1e-9:
            local_y = np.cross(np.array([0.0, 1.0, 0.0]), local_x)
        local_y = local_y / np.linalg.norm(local_y)
        local_z = np.cross(local_x, local_y)

    R = np.array([local_x, local_y, local_z])
    T_node_6x6 = np.zeros((6,6))
    T_node_6x6[:3, :3] = T_node_6x6[3:, 3:] = R
    T = np.zeros((12,12))
    T[:6, :6] = T[6:12, 6:12] = T_node_6x6

    # Transform global displacements to local element DOFs
    dofs_per_node = 6
    element_dofs = []
    for node_idx in element_global_indices:
        element_dofs.extend([node_idx * dofs_per_node + i for i in range(6)])
    
    u_local = T @ nodal_displacements[element_dofs]

    # Re-calculate local element forces
    Ke_local = np.zeros((12, 12))
    # Axial
    Ke_local[0,0] = Ke_local[6,6] = E*A/L; Ke_local[0,6] = Ke_local[6,0] = -E*A/L
    # Torsion
    Ke_local[3,3] = Ke_local[9,9] = G*J/L; Ke_local[3,9] = Ke_local[9,3] = -G*J/L
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

    f_local = Ke_local @ u_local

    # Max stress evaluated at node endpoints
    Fx2 = f_local[6]
    Mx2 = f_local[9]
    My2 = f_local[10]
    Mz2 = f_local[11]

    # Stresses evaluated at 4 outer cross-sectional points
    vm_list = []
    points = [[ro, 0.0], [-ro, 0.0], [0.0, ro], [0.0, -ro]]
    for yp, zp in points:
        sigma_axial = Fx2 / A
        sigma_bending_z = (-Mz2 * yp) / Iz
        sigma_bending_y = (My2 * zp) / Iy
        sigma_normal = sigma_axial + sigma_bending_z + sigma_bending_y
        
        tau_torsion = (np.abs(Mx2) * ro) / J
        
        von_mises = np.sqrt(sigma_normal**2 + 3 * tau_torsion**2)
        vm_list.append(von_mises)

    return {
        'local_forces_moments': f_local,
        'von_mises_stresses': {
            'y_plus': vm_list[0],
            'y_minus': vm_list[1],
            'z_plus': vm_list[2],
            'z_minus': vm_list[3],
            'max': max(vm_list)
        }
    }

element_results = {}
max_model_stress = 0.0
for element_idx, element_indices in enumerate(elements_3d):
    res = calculate_element_results_3d_beam(
        nodes_3d[element_indices[0]], nodes_3d[element_indices[1]],
        element_indices, U, material_properties, cross_section_properties
    )
    element_results[element_idx] = res
    if res['von_mises_stresses']['max'] > max_model_stress:
        max_model_stress = res['von_mises_stresses']['max']

# --- 6. Results Dashboard Metrics ---
col1, col2, col3 = st.columns(3)

# Calculated metrics
tip_dx = U[tip_node_idx*6 + 0]
tip_dy = U[tip_node_idx*6 + 1]
tip_dz = U[tip_node_idx*6 + 2]
total_deflection = np.sqrt(tip_dx**2 + tip_dy**2 + tip_dz**2)

safety_factor = yield_strength / max_model_stress if max_model_stress > 1e-3 else 999.0

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">💡 Total Tip Deflection</div>
        <div class="metric-val">{total_deflection * 1e3:.4f} mm</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🔥 Max Von Mises Stress</div>
        <div class="metric-val">{max_model_stress / 1e6:.3f} MPa</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    status_color = "#22c55e" if safety_factor >= 1.5 else ("#eab308" if safety_factor >= 1.0 else "#ef4444")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🛡️ Safety Factor</div>
        <div class="metric-val" style="color: {status_color}">{safety_factor:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

# Safety indicator warning
if safety_factor < 1.0:
    st.error(f"⚠️ STRUCTURAL FAILURE DETECTED: The structural elements have exceeded the material's yield strength ({yield_strength/1e6:.1f} MPa) by a factor of {1.0/safety_factor:.2f}x!")
elif safety_factor < 1.5:
    st.warning("⚠️ CRITICAL MARGIN: The factor of safety is below standard structural design guidelines (FoS < 1.5). Consider increasing section radii.")
else:
    st.success("✅ STRUCTURAL STABILITY SECURE: All spatial members satisfy the designated material factor of safety margins.")

# --- 7. Plotting Section ---
st.subheader("📊 3D Stress Mapping & Volumetric Deflection Plot")
st.markdown("Cylindrical tube mesh visualization displaying stress intensity gradient along the spatial model. Displacements are scaled for visual highlight.")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Setup equal axes layout boundaries
max_range = max(beam_length * 0.6, 0.5)
mid_x = beam_length * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

# Form deformed coordinates
displacements_reshaped = U.reshape(-1, 6)
deformed_nodes = nodes_3d + displacements_reshaped[:, :3] * scale_factor

# Mappings & Colormap configurations
all_stresses = [res['von_mises_stresses']['max'] for res in element_results.values()]
norm = plt.Normalize(min(all_stresses) - 1e3, max(all_stresses) + 1e3) if max(all_stresses) - min(all_stresses) < 1e-3 else plt.Normalize(min(all_stresses), max(all_stresses))
cmap = plt.cm.jet

resolution = 12

for i, (n1, n2) in enumerate(elements_3d):
    pt1 = deformed_nodes[n1]
    pt2 = deformed_nodes[n2]
    
    color = cmap(norm(all_stresses[i]))
    vector = pt2 - pt1
    mag = np.linalg.norm(vector)
    if mag < 1e-9: continue
    
    vector_norm = vector / mag
    
    # Cylindrical generator
    if np.isclose(np.abs(vector_norm[2]), 1.0):
        v1 = np.array([1.0, 0.0, 0.0])
    else:
        v1 = np.array([0.0, 0.0, 1.0])
        
    v1 = v1 - np.dot(v1, vector_norm) * vector_norm
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(vector_norm, v1)
    
    theta = np.linspace(0, 2*np.pi, resolution)
    circle_outer = outer_radius * (np.outer(np.cos(theta), v1) + np.outer(np.sin(theta), v2))
    circle_inner = inner_radius * (np.outer(np.cos(theta), v1) + np.outer(np.sin(theta), v2))
    
    verts_outer = []
    for j in range(resolution - 1):
        verts_outer.append([
            pt1 + circle_outer[j, :],
            pt1 + circle_outer[j+1, :],
            pt2 + circle_outer[j+1, :],
            pt2 + circle_outer[j, :]
        ])
    verts_outer.append([
        pt1 + circle_outer[resolution-1, :],
        pt1 + circle_outer[0, :],
        pt2 + circle_outer[0, :],
        pt2 + circle_outer[resolution-1, :]
    ])
    ax.add_collection3d(Poly3DCollection(verts_outer, facecolor=color, edgecolor='k', linewidths=0.15, alpha=0.9))

    if inner_radius > 0:
        verts_inner = []
        for j in range(resolution - 1):
            verts_inner.append([
                pt1 + circle_inner[j, :],
                pt1 + circle_inner[j+1, :],
                pt2 + circle_inner[j+1, :],
                pt2 + circle_inner[j, :]
            ])
        verts_inner.append([
            pt1 + circle_inner[resolution-1, :],
            pt1 + circle_inner[0, :],
            pt2 + circle_inner[0, :],
            pt2 + circle_inner[resolution-1, :]
        ])
        ax.add_collection3d(Poly3DCollection(verts_inner, facecolor='gray', edgecolor='k', linewidths=0.1, alpha=0.4))

# Plot reference nodes and labels
ax.scatter(deformed_nodes[:, 0], deformed_nodes[:, 1], deformed_nodes[:, 2], color='black', s=80, zorder=5)
for idx, pt in enumerate(deformed_nodes):
    ax.text(pt[0]+0.02, pt[1]+0.02, pt[2]+0.02, f"Node {idx}", color='black', fontweight='bold', fontsize=8)

ax.set_xlabel('X Coordinate (m)')
ax.set_ylabel('Y Coordinate (m)')
ax.set_zlabel('Z Coordinate (m)')

# Render Colorbar inside Streamlit
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array(all_stresses)
cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
cbar.set_label('Von Mises Equivalent Stress (Pa)')

st.pyplot(fig)

# --- 8. Tabular Nodal & Element Reports ---
st.subheader("📋 Structural Model Logs & Dataframes")
tab1, tab2 = st.tabs(["Nodal Displacements", "Element Internal Forces & Stresses"])

with tab1:
    nodal_data = []
    for i in range(num_nodes):
        node_disp = U[i*6:(i+1)*6]
        nodal_data.append({
            "Node ID": i,
            "Disp X (mm)": node_disp[0]*1e3,
            "Disp Y (mm)": node_disp[1]*1e3,
            "Disp Z (mm)": node_disp[2]*1e3,
            "Rotation Rx (mrad)": node_disp[3]*1e3,
            "Rotation Ry (mrad)": node_disp[4]*1e3,
            "Rotation Rz (mrad)": node_disp[5]*1e3
        })
    st.dataframe(pd.DataFrame(nodal_data), use_container_width=True)

with tab2:
    element_data = []
    for element_idx, res in element_results.items():
        forces = res['local_forces_moments']
        max_vm = res['von_mises_stresses']['max']
        element_data.append({
            "Element ID": element_idx,
            "Connectivity": f"Node {elements_3d[element_idx][0]} ➡️ {elements_3d[element_idx][1]}",
            "Axial Force Fx (N)": forces[6],
            "Torsional Moment Mx (N-m)": forces[9],
            "Bending Moment My (N-m)": forces[10],
            "Bending Moment Mz (N-m)": forces[11],
            "Max Von Mises (MPa)": max_vm / 1e6
        })
    st.dataframe(pd.DataFrame(element_data), use_container_width=True)


