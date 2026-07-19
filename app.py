import customtkinter as ctk
import tkinter.ttk as ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.linalg import solve

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# --- MATH FUNCTIONS ---
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
    Ke_local[0,0] = Ke_local[6,6] = E*A/L; Ke_local[0,6] = Ke_local[6,0] = -E*A/L
    Ke_local[3,3] = Ke_local[9,9] = G*J/L; Ke_local[3,9] = Ke_local[9,3] = -G*J/L
    bz1, bz2, bz3, bz4 = 12*E*Iz/L**3, 6*E*Iz/L**2, 4*E*Iz/L, 2*E*Iz/L
    Ke_local[1,1] = Ke_local[7,7] = bz1; Ke_local[1,7] = Ke_local[7,1] = -bz1
    Ke_local[1,5] = Ke_local[5,1] = Ke_local[1,11] = Ke_local[11,1] = bz2
    Ke_local[5,5] = Ke_local[11,11] = bz3; Ke_local[5,11] = Ke_local[11,5] = bz4
    Ke_local[5,7] = Ke_local[7,5] = Ke_local[7,11] = Ke_local[11,7] = -bz2
    by1, by2, by3, by4 = 12*E*Iy/L**3, 6*E*Iy/L**2, 4*E*Iy/L, 2*E*Iy/L
    Ke_local[2,2] = Ke_local[8,8] = by1; Ke_local[2,8] = Ke_local[8,2] = -by1
    Ke_local[2,4] = Ke_local[4,2] = Ke_local[2,10] = Ke_local[10,2] = -by2
    Ke_local[4,4] = Ke_local[10,10] = by3; Ke_local[4,10] = Ke_local[10,4] = by4
    Ke_local[4,8] = Ke_local[8,4] = Ke_local[8,10] = Ke_local[10,8] = by2

    f_local = Ke_local @ u_local

    Fx2 = f_local[6]
    Mx2 = f_local[9]
    My2 = f_local[10]
    Mz2 = f_local[11]

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

# --- DOCUMENTATION ---
DOC_TEXTS = {
    "Welcome": "Welcome to the 3D FEA Engine!\n\nHover over any configuration panel on the left to see its documentation here.",
    "Material": "### Material Selection\nDefines the isotropic material properties of the structure.\n\nE: Young's Modulus (stiffness).\nν: Poisson's Ratio (lateral strain).\nYield Strength: Material's limit before plastic deformation.\n\nPresets load values automatically, or choose 'Custom Isotropic' to enter your own.",
    "Geometry": "### Beam Geometry\nConfigures the physical dimensions of the spatial hollow tube elements.\n\nOuter/Inner Radius: Dictates the cross-sectional area and moments of inertia. Ensure Outer Radius > Inner Radius.\n\nLength: Total horizontal span of the beam.\nElements: Number of finite elements the beam is discretized into for higher resolution.",
    "Loads": "### Tip Load Conditions\nDefines the external forces and moments applied at the tip node (the free right end).\n\nFx: Axial Force (Tension/Compression)\nFy: Transverse Shear (Vertical)\nFz: Transverse Shear (Lateral/Depth)\nMx: Torsional Moment around the X-axis",
    "Scale": "### Deflection Scale\nA visual multiplier applied only to the 3D plot to exaggerate structural bending for easier visual inspection. This does not change the actual computed displacements.",
    "Results": "### Stress & Displacement Results\nDisplays the calculated 6-DOF nodal displacements and element internal forces.\n\nThe Von Mises Stress determines structural safety based on combined axial, bending, and torsional loads."
}

# --- GUI APP ---
class FEAEngineApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("3D Beam FEA Engine")
        self.geometry("1600x900")
        
        # Grid config
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # State variables
        self.fig = None
        self.canvas = None
        
        self.setup_left_panel()
        self.setup_mid_panel()
        self.setup_right_panel()
        
        self.update_doc("Welcome")
        self.run_analysis()

    def update_doc(self, key):
        self.doc_textbox.configure(state="normal")
        self.doc_textbox.delete("0.0", "end")
        self.doc_textbox.insert("0.0", DOC_TEXTS.get(key, ""))
        self.doc_textbox.configure(state="disabled")

    def on_material_change(self, choice):
        if choice == "Structural Steel":
            self.e_var.set(200.0)
            self.pr_var.set(0.27)
            self.yield_var.set(250.0)
            self._set_mat_state("disabled")
        elif choice == "Aluminum 6061-T6":
            self.e_var.set(68.9)
            self.pr_var.set(0.33)
            self.yield_var.set(276.0)
            self._set_mat_state("disabled")
        elif choice == "Titanium Grade 5":
            self.e_var.set(114.0)
            self.pr_var.set(0.34)
            self.yield_var.set(880.0)
            self._set_mat_state("disabled")
        else:
            self._set_mat_state("normal")
            
    def _set_mat_state(self, state):
        self.e_entry.configure(state=state)
        self.pr_entry.configure(state=state)
        self.yield_entry.configure(state=state)

    def setup_left_panel(self):
        self.left_panel = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.left_panel.grid(row=0, column=0, sticky="nsew")
        self.left_panel.grid_rowconfigure(10, weight=1)
        
        ctk.CTkLabel(self.left_panel, text="⚙️ Configuration", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Material
        ctk.CTkLabel(self.left_panel, text="💎 Material Selection", anchor="w").grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.material_var = ctk.StringVar(value="Structural Steel")
        self.material_dropdown = ctk.CTkComboBox(self.left_panel, variable=self.material_var, values=["Structural Steel", "Aluminum 6061-T6", "Titanium Grade 5", "Custom Isotropic"], command=self.on_material_change)
        self.material_dropdown.grid(row=2, column=0, padx=20, pady=(5, 10), sticky="ew")
        
        self.e_var = ctk.DoubleVar(value=200.0)
        self.pr_var = ctk.DoubleVar(value=0.27)
        self.yield_var = ctk.DoubleVar(value=250.0)
        
        self.mat_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.mat_frame.grid(row=3, column=0, padx=20, sticky="ew")
        self.mat_frame.columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self.mat_frame, text="E (GPa):").grid(row=0, column=0, sticky="w")
        self.e_entry = ctk.CTkEntry(self.mat_frame, textvariable=self.e_var, state="disabled", width=80)
        self.e_entry.grid(row=0, column=1, sticky="e")
        
        ctk.CTkLabel(self.mat_frame, text="Poisson's:").grid(row=1, column=0, sticky="w")
        self.pr_entry = ctk.CTkEntry(self.mat_frame, textvariable=self.pr_var, state="disabled", width=80)
        self.pr_entry.grid(row=1, column=1, sticky="e")
        
        ctk.CTkLabel(self.mat_frame, text="Yield (MPa):").grid(row=2, column=0, sticky="w")
        self.yield_entry = ctk.CTkEntry(self.mat_frame, textvariable=self.yield_var, state="disabled", width=80)
        self.yield_entry.grid(row=2, column=1, sticky="e")
        
        self.material_dropdown.bind("<Enter>", lambda e: self.update_doc("Material"))
        self.mat_frame.bind("<Enter>", lambda e: self.update_doc("Material"))
        
        # Geometry
        ctk.CTkLabel(self.left_panel, text="📐 Beam Geometry", anchor="w").grid(row=4, column=0, padx=20, pady=(20, 0), sticky="w")
        
        self.geom_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.geom_frame.grid(row=5, column=0, padx=20, sticky="ew")
        self.geom_frame.bind("<Enter>", lambda e: self.update_doc("Geometry"))
        self.geom_frame.columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self.geom_frame, text="Outer Rad (m):").grid(row=0, column=0, sticky="w", pady=(0,10))
        self.or_var = ctk.DoubleVar(value=0.10)
        self.or_slider = ctk.CTkSlider(self.geom_frame, variable=self.or_var, from_=0.02, to=0.50)
        self.or_slider.grid(row=0, column=1, sticky="e", pady=(0,10))
        
        ctk.CTkLabel(self.geom_frame, text="Inner Rad (m):").grid(row=1, column=0, sticky="w", pady=(0,10))
        self.ir_var = ctk.DoubleVar(value=0.08)
        self.ir_slider = ctk.CTkSlider(self.geom_frame, variable=self.ir_var, from_=0.01, to=0.49)
        self.ir_slider.grid(row=1, column=1, sticky="e", pady=(0,10))
        
        ctk.CTkLabel(self.geom_frame, text="Length (m):").grid(row=2, column=0, sticky="w", pady=(0,10))
        self.len_var = ctk.DoubleVar(value=2.0)
        self.len_slider = ctk.CTkSlider(self.geom_frame, variable=self.len_var, from_=0.5, to=10.0)
        self.len_slider.grid(row=2, column=1, sticky="e", pady=(0,10))
        
        ctk.CTkLabel(self.geom_frame, text="Elements:").grid(row=3, column=0, sticky="w")
        self.el_var = ctk.IntVar(value=2)
        self.el_slider = ctk.CTkSlider(self.geom_frame, variable=self.el_var, from_=1, to=10, number_of_steps=9)
        self.el_slider.grid(row=3, column=1, sticky="e")
        
        # Loads
        ctk.CTkLabel(self.left_panel, text="⚡ Tip Load Conditions", anchor="w").grid(row=6, column=0, padx=20, pady=(20, 0), sticky="w")
        self.loads_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.loads_frame.grid(row=7, column=0, padx=20, sticky="ew")
        self.loads_frame.columnconfigure(1, weight=1)
        self.loads_frame.bind("<Enter>", lambda e: self.update_doc("Loads"))
        
        ctk.CTkLabel(self.loads_frame, text="Fx (N):").grid(row=0, column=0, sticky="w")
        self.fx_var = ctk.DoubleVar(value=0.0)
        ctk.CTkEntry(self.loads_frame, textvariable=self.fx_var, width=80).grid(row=0, column=1, sticky="e", pady=2)
        
        ctk.CTkLabel(self.loads_frame, text="Fy (N):").grid(row=1, column=0, sticky="w")
        self.fy_var = ctk.DoubleVar(value=-1000.0)
        ctk.CTkEntry(self.loads_frame, textvariable=self.fy_var, width=80).grid(row=1, column=1, sticky="e", pady=2)
        
        ctk.CTkLabel(self.loads_frame, text="Fz (N):").grid(row=2, column=0, sticky="w")
        self.fz_var = ctk.DoubleVar(value=0.0)
        ctk.CTkEntry(self.loads_frame, textvariable=self.fz_var, width=80).grid(row=2, column=1, sticky="e", pady=2)
        
        ctk.CTkLabel(self.loads_frame, text="Mx (N-m):").grid(row=3, column=0, sticky="w")
        self.mx_var = ctk.DoubleVar(value=0.0)
        ctk.CTkEntry(self.loads_frame, textvariable=self.mx_var, width=80).grid(row=3, column=1, sticky="e", pady=2)
        
        # Scale
        scale_label = ctk.CTkLabel(self.left_panel, text="🎨 Deflection Scale")
        scale_label.grid(row=8, column=0, padx=20, pady=(20, 0), sticky="w")
        scale_label.bind("<Enter>", lambda e: self.update_doc("Scale"))
        self.scale_var = ctk.DoubleVar(value=50.0)
        self.scale_slider = ctk.CTkSlider(self.left_panel, variable=self.scale_var, from_=1.0, to=500.0)
        self.scale_slider.grid(row=9, column=0, padx=20, sticky="ew")
        self.scale_slider.bind("<Enter>", lambda e: self.update_doc("Scale"))
        
        # Button
        self.run_btn = ctk.CTkButton(self.left_panel, text="Run Analysis", command=self.run_analysis)
        self.run_btn.grid(row=11, column=0, padx=20, pady=20, sticky="ew")
        
    def setup_mid_panel(self):
        self.mid_panel = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.mid_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.mid_panel.grid_rowconfigure(1, weight=1)
        self.mid_panel.grid_columnconfigure(0, weight=1)
        
        # Metrics Top bar
        self.metrics_frame = ctk.CTkFrame(self.mid_panel, fg_color="transparent")
        self.metrics_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.metrics_frame.grid_columnconfigure((0,1,2), weight=1)
        
        # Metric Cards
        self.m1 = ctk.CTkFrame(self.metrics_frame, corner_radius=8, fg_color="#1e2942")
        self.m1.grid(row=0, column=0, sticky="ew", padx=5)
        ctk.CTkLabel(self.m1, text="💡 Total Tip Deflection", text_color="#9ca3af").pack(pady=(10, 0))
        self.deflection_lbl = ctk.CTkLabel(self.m1, text="0.000 mm", font=ctk.CTkFont(size=24, weight="bold"), text_color="#06b6d4")
        self.deflection_lbl.pack(pady=(0, 10))
        
        self.m2 = ctk.CTkFrame(self.metrics_frame, corner_radius=8, fg_color="#1e2942")
        self.m2.grid(row=0, column=1, sticky="ew", padx=5)
        ctk.CTkLabel(self.m2, text="🔥 Max Von Mises Stress", text_color="#9ca3af").pack(pady=(10, 0))
        self.stress_lbl = ctk.CTkLabel(self.m2, text="0.000 MPa", font=ctk.CTkFont(size=24, weight="bold"), text_color="#06b6d4")
        self.stress_lbl.pack(pady=(0, 10))
        
        self.m3 = ctk.CTkFrame(self.metrics_frame, corner_radius=8, fg_color="#1e2942")
        self.m3.grid(row=0, column=2, sticky="ew", padx=5)
        ctk.CTkLabel(self.m3, text="🛡️ Safety Factor", text_color="#9ca3af").pack(pady=(10, 0))
        self.sf_lbl = ctk.CTkLabel(self.m3, text="0.00", font=ctk.CTkFont(size=24, weight="bold"))
        self.sf_lbl.pack(pady=(0, 10))
        
        # Plot Frame
        self.plot_frame = ctk.CTkFrame(self.mid_panel, corner_radius=8)
        self.plot_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        
        # Table Tabs
        self.tabview = ctk.CTkTabview(self.mid_panel, height=250)
        self.tabview.grid(row=2, column=0, sticky="ew")
        self.tabview.add("Nodal Displacements")
        self.tabview.add("Element Internal Forces")
        
        # Use treeviews for data
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b", borderwidth=0)
        style.map('Treeview', background=[('selected', '#1f538d')])
        
        self.node_tree = ttk.Treeview(self.tabview.tab("Nodal Displacements"), columns=("ID", "DX", "DY", "DZ", "RX", "RY", "RZ"), show="headings")
        self.node_tree.heading("ID", text="Node ID")
        self.node_tree.heading("DX", text="Disp X (mm)")
        self.node_tree.heading("DY", text="Disp Y (mm)")
        self.node_tree.heading("DZ", text="Disp Z (mm)")
        self.node_tree.heading("RX", text="Rot X (mrad)")
        self.node_tree.heading("RY", text="Rot Y (mrad)")
        self.node_tree.heading("RZ", text="Rot Z (mrad)")
        self.node_tree.pack(fill="both", expand=True)
        
        self.elem_tree = ttk.Treeview(self.tabview.tab("Element Internal Forces"), columns=("ID", "Conn", "Fx", "Mx", "My", "Mz", "VM"), show="headings")
        self.elem_tree.heading("ID", text="Element ID")
        self.elem_tree.heading("Conn", text="Connectivity")
        self.elem_tree.heading("Fx", text="Axial Fx (N)")
        self.elem_tree.heading("Mx", text="Torsion Mx (N-m)")
        self.elem_tree.heading("My", text="Bending My (N-m)")
        self.elem_tree.heading("Mz", text="Bending Mz (N-m)")
        self.elem_tree.heading("VM", text="Max Von Mises (MPa)")
        self.elem_tree.pack(fill="both", expand=True)

    def setup_right_panel(self):
        self.right_panel = ctk.CTkFrame(self, width=350, corner_radius=0, fg_color="#1a1a1a")
        self.right_panel.grid(row=0, column=2, sticky="nsew")
        self.right_panel.grid_rowconfigure(1, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(self.right_panel, text="📖 Documentation", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        self.doc_textbox = ctk.CTkTextbox(self.right_panel, wrap="word", corner_radius=8, fg_color="#242424")
        self.doc_textbox.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.doc_textbox.insert("0.0", "Welcome to the FEA Engine.")
        self.doc_textbox.configure(state="disabled")

    def run_analysis(self):
        # 1. Gather properties
        youngs_modulus = self.e_var.get() * 1e9
        poissons_ratio = self.pr_var.get()
        yield_strength = self.yield_var.get() * 1e6
        
        outer_radius = self.or_var.get()
        inner_radius = self.ir_var.get()
        if inner_radius >= outer_radius:
            inner_radius = outer_radius - 0.001
            self.ir_var.set(inner_radius)
            
        beam_length = self.len_var.get()
        num_elements = self.el_var.get()
        
        fx_load = self.fx_var.get()
        fy_load = self.fy_var.get()
        fz_load = self.fz_var.get()
        mx_load = self.mx_var.get()
        scale_factor = self.scale_var.get()
        
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
        
        # 2. Build 3D Mesh
        x_coords = np.linspace(0.0, beam_length, num_elements + 1)
        nodes_3d = np.zeros((num_elements + 1, 3))
        nodes_3d[:, 0] = x_coords
        
        elements_3d = []
        for i in range(num_elements):
            elements_3d.append([i, i + 1])
        elements_3d = np.array(elements_3d)
        
        # 3. Global System Assembly & Solution
        num_nodes = len(nodes_3d)
        K = np.zeros((num_nodes*6, num_nodes*6))
        F = np.zeros(num_nodes*6)
        
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
                    
        fixed_dofs = np.arange(0, 6)
        for dof in fixed_dofs:
            K[dof, :] = 0; K[:, dof] = 0; K[dof, dof] = 1; F[dof] = 0
            
        U = solve(K, F)
        
        # 4. Stress Post-Processing
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
                
        # 5. Dashboard Metrics Update
        tip_dx = U[tip_node_idx*6 + 0]
        tip_dy = U[tip_node_idx*6 + 1]
        tip_dz = U[tip_node_idx*6 + 2]
        total_deflection = np.sqrt(tip_dx**2 + tip_dy**2 + tip_dz**2)
        
        safety_factor = yield_strength / max_model_stress if max_model_stress > 1e-3 else 999.0
        
        self.deflection_lbl.configure(text=f"{total_deflection * 1e3:.4f} mm")
        self.stress_lbl.configure(text=f"{max_model_stress / 1e6:.3f} MPa")
        self.sf_lbl.configure(text=f"{safety_factor:.2f}")
        if safety_factor >= 1.5:
            self.sf_lbl.configure(text_color="#22c55e")
        elif safety_factor >= 1.0:
            self.sf_lbl.configure(text_color="#eab308")
        else:
            self.sf_lbl.configure(text_color="#ef4444")
            
        # Update Tables
        for row in self.node_tree.get_children():
            self.node_tree.delete(row)
        for i in range(num_nodes):
            nd = U[i*6:(i+1)*6]
            self.node_tree.insert("", "end", values=(i, f"{nd[0]*1e3:.4f}", f"{nd[1]*1e3:.4f}", f"{nd[2]*1e3:.4f}", f"{nd[3]*1e3:.4f}", f"{nd[4]*1e3:.4f}", f"{nd[5]*1e3:.4f}"))
            
        for row in self.elem_tree.get_children():
            self.elem_tree.delete(row)
        for element_idx, res in element_results.items():
            f = res['local_forces_moments']
            self.elem_tree.insert("", "end", values=(element_idx, f"{elements_3d[element_idx][0]} -> {elements_3d[element_idx][1]}", f"{f[6]:.2f}", f"{f[9]:.2f}", f"{f[10]:.2f}", f"{f[11]:.2f}", f"{res['von_mises_stresses']['max']/1e6:.2f}"))

        # 6. Matplotlib 3D Update
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(8, 6), facecolor='#2b2b2b')
        ax = self.fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#2b2b2b')
        
        max_range = max(beam_length * 0.6, 0.5)
        mid_x = beam_length * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        displacements_reshaped = U.reshape(-1, 6)
        deformed_nodes = nodes_3d + displacements_reshaped[:, :3] * scale_factor
        
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
            if np.isclose(np.abs(vector_norm[2]), 1.0):
                v1 = np.array([1.0, 0.0, 0.0])
            else:
                v1 = np.array([0.0, 0.0, 1.0])
                
            v1 = v1 - np.dot(v1, vector_norm) * vector_norm
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(vector_norm, v1)
            
            theta = np.linspace(0, 2*np.pi, resolution)
            circle_outer = outer_radius * (np.outer(np.cos(theta), v1) + np.outer(np.sin(theta), v2))
            
            verts_outer = []
            for j in range(resolution - 1):
                verts_outer.append([
                    pt1 + circle_outer[j, :], pt1 + circle_outer[j+1, :],
                    pt2 + circle_outer[j+1, :], pt2 + circle_outer[j, :]
                ])
            verts_outer.append([
                pt1 + circle_outer[resolution-1, :], pt1 + circle_outer[0, :],
                pt2 + circle_outer[0, :], pt2 + circle_outer[resolution-1, :]
            ])
            ax.add_collection3d(Poly3DCollection(verts_outer, facecolor=color, edgecolor='k', linewidths=0.15, alpha=0.9))

        ax.scatter(deformed_nodes[:, 0], deformed_nodes[:, 1], deformed_nodes[:, 2], color='white', s=80, zorder=5)
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_zlabel('Z Coordinate (m)')
        
        # Colorbar
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(all_stresses)
        cbar = self.fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
        cbar.set_label('Von Mises Equivalent Stress (Pa)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    app = FEAEngineApp()
    app.mainloop()
