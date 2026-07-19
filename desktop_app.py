import sys
import os
import re
import numpy as np
import pandas as pd
from scipy.linalg import solve
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QStackedWidget, QSlider, QDoubleSpinBox, 
    QSpinBox, QComboBox, QTextEdit, QTableWidget, QTableWidgetItem,
    QTabWidget, QHeaderView, QGroupBox, QGridLayout, QSplitter, QFileDialog
)
from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtGui import QFont, QColor, QPalette

# WNFEA Core Imports
from wnfea.model import FEAModel, PropertyAssignment
from wnfea.geometry.primitives import Point3D, GeometryNode, GeometryEdge
from wnfea.geometry.step_parser import STEPParser
from wnfea.properties.materials import MaterialDef
from wnfea.properties.sections import SectionDef
from wnfea.mesh.beam_mesher import BeamMesher
from wnfea.solver.linear_static import solve_linear_static
from wnfea.solver.stress import compute_element_stresses
from wnfea.results.result_set import ResultSet

# --- Conversion Constants ---
MM_TO_IN = 1.0 / 25.4
IN_TO_MM = 25.4

N_TO_LBF = 0.22480894
LBF_TO_N = 4.4482216

MPA_TO_PSI = 145.0377377
PSI_TO_MPA = 1.0 / 145.0377377

TONNE_TO_LBM = 2204.62262
LBM_TO_TONNE = 1.0 / 2204.62262

TONNE_MM3_TO_LBM_IN3 = 3.6127292e7
LBM_IN3_TO_TONNE_MM3 = 1.0 / TONNE_MM3_TO_LBM_IN3

NMM_TO_LBFIN = N_TO_LBF * MM_TO_IN
LBFIN_TO_NMM = LBF_TO_N * IN_TO_MM

# --- STEP Parsing Utilities ---
def parse_step_file(file_path):
    """
    Parses a STEP (.stp/.step) file to extract linear curve segments (edges).
    Returns a list of tuples: (pt1, pt2) where each point is a 3D numpy array.
    """
    cartesian_points = {}
    vertex_points = {}
    edges = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
        
    # 1. Parse CARTESIAN_POINT
    cp_pattern = re.compile(
        r'#(\d+)\s*=\s*CARTESIAN_POINT\s*\(\s*[^,]*\s*,\s*\(\s*([^)]+)\s*\)\s*\)', 
        re.IGNORECASE
    )
    for match in cp_pattern.finditer(content):
        cp_id = int(match.group(1))
        coords_str = match.group(2)
        try:
            coords = [float(x.strip()) for x in coords_str.split(',')]
            if len(coords) >= 3:
                cartesian_points[cp_id] = np.array(coords[:3])
            elif len(coords) == 2:
                cartesian_points[cp_id] = np.array([coords[0], coords[1], 0.0])
        except ValueError:
            continue

    # 2. Parse VERTEX_POINT
    vp_pattern = re.compile(
        r'#(\d+)\s*=\s*VERTEX_POINT\s*\(\s*[^,]*\s*,\s*#(\d+)\s*\)', 
        re.IGNORECASE
    )
    for match in vp_pattern.finditer(content):
        vp_id = int(match.group(1))
        cp_id = int(match.group(2))
        vertex_points[vp_id] = cp_id

    # 3. Parse EDGE_CURVE
    ec_pattern = re.compile(
        r'#(\d+)\s*=\s*EDGE_CURVE\s*\(\s*[^,]*\s*,\s*#(\d+)\s*,\s*#(\d+)\s*,', 
        re.IGNORECASE
    )
    for match in ec_pattern.finditer(content):
        vp1_id = int(match.group(2))
        vp2_id = int(match.group(3))
        
        cp1_id = vertex_points.get(vp1_id)
        cp2_id = vertex_points.get(vp2_id)
        
        if cp1_id in cartesian_points and cp2_id in cartesian_points:
            edges.append((cartesian_points[cp1_id], cartesian_points[cp2_id]))
            
    # 4. Parse POLYLINE (fallback for wireframe models)
    pl_pattern = re.compile(
        r'#(\d+)\s*=\s*POLYLINE\s*\(\s*[^,]*\s*,\s*\(\s*([^)]+)\s*\)\s*\)', 
        re.IGNORECASE
    )
    for match in pl_pattern.finditer(content):
        points_list_str = match.group(2)
        pt_ids = [int(x.strip().replace('#', '')) for x in points_list_str.split(',') if '#' in x]
        for i in range(len(pt_ids) - 1):
            cp1_id = pt_ids[i]
            cp2_id = pt_ids[i+1]
            if cp1_id in cartesian_points and cp2_id in cartesian_points:
                edges.append((cartesian_points[cp1_id], cartesian_points[cp2_id]))
                
    return edges

def consolidate_edges(edges, tolerance=1e-5):
    """
    Consolidates line segments into unique nodes and elements based on tolerance.
    """
    unique_nodes = []
    elements = []
    
    def get_node_index(pt):
        for idx, node in enumerate(unique_nodes):
            if np.linalg.norm(node - pt) < tolerance:
                return idx
        unique_nodes.append(pt)
        return len(unique_nodes) - 1
        
    for pt1, pt2 in edges:
        idx1 = get_node_index(pt1)
        idx2 = get_node_index(pt2)
        if idx1 != idx2:
            elements.append([idx1, idx2])
            
    return np.array(unique_nodes), np.array(elements)

# --- Custom Styling (Premium Dark Mode) ---
STYLESHEET = """
QMainWindow {
    background-color: #0b0f19;
}
QWidget {
    color: #f3f4f6;
    font-family: "Outfit", "Segoe UI", sans-serif;
}
/* Sidebar */
#Sidebar {
    background-color: #131a2b;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}
QPushButton#NavBtn {
    background-color: transparent;
    color: #9ca3af;
    text-align: left;
    padding: 15px;
    font-size: 14px;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    margin: 5px;
}
QPushButton#NavBtn:hover {
    background-color: #1e2942;
    color: #ffffff;
}
QPushButton#NavBtn:checked {
    background-color: #1e2942;
    color: #06b6d4;
    border-left: 4px solid #06b6d4;
}
/* Panels & GroupBoxes */
QGroupBox {
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    margin-top: 20px;
    background-color: #1e2942;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    color: #06b6d4;
    font-weight: bold;
}
QLabel#MetricTitle {
    color: #9ca3af;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QLabel#MetricValue {
    font-size: 24px;
    font-weight: bold;
    color: #06b6d4;
}
/* Inputs */
QSlider::groove:horizontal {
    border: 1px solid #131a2b;
    height: 8px;
    background: #0b0f19;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #06b6d4;
    width: 18px;
    margin: -5px 0;
    border-radius: 9px;
}
QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #0b0f19;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 5px;
    border-radius: 4px;
    color: #ffffff;
}
QPushButton#RunBtn {
    background-color: #06b6d4;
    color: #000000;
    font-weight: bold;
    font-size: 16px;
    padding: 15px;
    border-radius: 8px;
}
QPushButton#RunBtn:hover {
    background-color: #0891b2;
}
QTextEdit {
    background-color: #050811;
    color: #e5e7eb;
    border: 1px solid rgba(255, 255, 255, 0.1);
    font-family: "Fira Code", "Consolas", monospace;
    border-radius: 8px;
}
QTableWidget {
    background-color: #131a2b;
    alternate-background-color: #1e2942;
    border: 1px solid rgba(255, 255, 255, 0.1);
    gridline-color: rgba(255, 255, 255, 0.05);
}
QHeaderView::section {
    background-color: #0b0f19;
    color: #06b6d4;
    padding: 5px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
"""

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = plt.figure(figsize=(8, 6), facecolor='#0b0f19')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#0b0f19')
        # Dark theme axes
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#1e2942')
        self.ax.yaxis.pane.set_edgecolor('#1e2942')
        self.ax.zaxis.pane.set_edgecolor('#1e2942')
        self.ax.tick_params(colors='#9ca3af')
        self.ax.xaxis.label.set_color('#9ca3af')
        self.ax.yaxis.label.set_color('#9ca3af')
        self.ax.zaxis.label.set_color('#9ca3af')
        super().__init__(self.fig)
        self.setParent(parent)

class FEAEngineApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏗️ AERO-STRUCT 3D FEA Engine")
        self.resize(1200, 800)
        self.setStyleSheet(STYLESHEET)
        
        # State variables
        self.unit_system = "Metric"
        self.nodes_3d = None
        self.elements_3d = None
        self.U = None
        self.element_results = {}
        self.max_model_stress = 0.0
        self.total_deflection = 0.0
        self.safety_factor = 0.0
        
        # Central FEA Model
        self.model = FEAModel()
        
        # Generalized Cross-Section Properties (in mm, mm^2, mm^4)
        self.A = 11300.0
        self.Iy = 4.64e7
        self.Iz = 4.64e7
        self.J = 9.28e7
        self.y_max = 100.0
        self.z_max = 100.0
        
        # STEP Geometry specific state
        self.step_edges = None
        self.imported_nodes = None
        self.imported_elements = None
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)
        
        # --- Left Sidebar ---
        self.sidebar = QWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(250)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)
        sidebar_layout.setSpacing(5)
        
        title_lbl = QLabel("AERO-STRUCT v1.2")
        title_lbl.setStyleSheet("color: #06b6d4; font-size: 18px; font-weight: bold; margin-bottom: 20px;")
        title_lbl.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title_lbl)
        
        self.nav_buttons = []
        nav_items = ["📐 Geometry", "💠 Meshing", "⚡ Problem Setup", "⚙️ Solving", "📊 Results"]
        
        for i, item in enumerate(nav_items):
            btn = QPushButton(item)
            btn.setObjectName("NavBtn")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self.switch_page(idx))
            sidebar_layout.addWidget(btn)
            self.nav_buttons.append(btn)
            
        sidebar_layout.addStretch()
        
        # Unit System Dropdown
        self.lbl_unit_dropdown_title = QLabel("📐 Unit System:")
        self.lbl_unit_dropdown_title.setStyleSheet("color: #9ca3af; font-size: 11px; margin-left: 10px; font-weight: bold;")
        sidebar_layout.addWidget(self.lbl_unit_dropdown_title)
        
        self.combo_units = QComboBox()
        self.combo_units.addItems([
            "N / mm / tonne (Metric)",
            "in / lbf / lbm (Imperial)"
        ])
        self.combo_units.setStyleSheet("margin: 5px; background-color: #0b0f19; border: 1px solid rgba(255, 255, 255, 0.1); color: #ffffff;")
        sidebar_layout.addWidget(self.combo_units)
        
        # Toggle Documentation Button
        self.btn_toggle_doc = QPushButton("📖 Toggle Docs")
        self.btn_toggle_doc.setObjectName("NavBtn")
        self.btn_toggle_doc.setCheckable(True)
        self.btn_toggle_doc.setChecked(True)
        self.btn_toggle_doc.clicked.connect(self.toggle_doc_panel)
        sidebar_layout.addWidget(self.btn_toggle_doc)
        
        self.splitter.addWidget(self.sidebar)
        
        # --- Middle Splitter (Controls + Viewport) ---
        self.middle_splitter = QSplitter(Qt.Horizontal)
        
        # Stacked widget (for inputs and controls only)
        self.stacked_widget = QStackedWidget()
        self.middle_splitter.addWidget(self.stacked_widget)
        
        # 3D Viewport Panel (Persistent on the right)
        self.viewport_container = QGroupBox("3D Viewport")
        viewport_layout = QVBoxLayout(self.viewport_container)
        viewport_layout.setContentsMargins(10, 10, 10, 10)
        
        self.geom_canvas = MatplotlibCanvas(self)
        self.canvas = self.geom_canvas
        self.geom_toolbar = NavigationToolbar(self.geom_canvas, self)
        self.geom_toolbar.setStyleSheet("background-color: #1e2942; color: #ffffff; border: none;")
        
        viewport_layout.addWidget(self.geom_toolbar)
        viewport_layout.addWidget(self.geom_canvas)
        
        self.middle_splitter.addWidget(self.viewport_container)
        self.middle_splitter.setStretchFactor(0, 4)
        self.middle_splitter.setStretchFactor(1, 6)
        
        self.splitter.addWidget(self.middle_splitter)
        
        # --- Right Documentation Panel ---
        self.doc_view = QWebEngineView()
        doc_settings = self.doc_view.settings()
        doc_settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        doc_settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        doc_path = os.path.join(base_dir, "index.html")
        self.doc_view.setUrl(QUrl.fromLocalFile(doc_path))
        self.splitter.addWidget(self.doc_view)
        
        # Set layout proportions
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 7)
        self.splitter.setStretchFactor(2, 3)
        
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setCollapsible(2, True)
        
        self.create_geometry_page()
        self.create_meshing_page()
        self.create_setup_page()
        self.create_solving_page()
        self.create_results_page()
        
        # Draw initial geometry preview
        self.update_visualization()
        
        # Select first page
        self.switch_page(0)
        
    def switch_page(self, index):
        self.stacked_widget.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        if index == 2:
            self.refresh_setup_nodes()
        self.update_visualization()

    def toggle_doc_panel(self):
        visible = self.btn_toggle_doc.isChecked()
        self.doc_view.setVisible(visible)

    # --- Pages ---
    def create_geometry_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        
        header = QLabel("📐 Geometry Setup")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        layout.addWidget(header)
        
        # 1. Section Properties Group (Dropdown + Dynamic stacked inputs in mm)
        self.sec_grp = QGroupBox("Beam Section Properties")
        sec_layout = QVBoxLayout(self.sec_grp)
        
        sec_layout.addWidget(QLabel("Select Cross-Section Shape:"))
        self.combo_shape = QComboBox()
        self.combo_shape.addItems([
            "Round Tube", 
            "Square Tube", 
            "Rectangular Tube", 
            "I-Beam", 
            "Round Bar (Solid)", 
            "Rectangular Bar (Solid)", 
            "General (Custom)"
        ])
        sec_layout.addWidget(self.combo_shape)
        
        # Stacked inputs widget
        self.shape_inputs_stacked = QStackedWidget()
        
        # Shape 0: Round Tube
        rt_widget = QWidget()
        rt_layout = QGridLayout(rt_widget)
        rt_layout.setContentsMargins(0, 5, 0, 5)
        self.lbl_rt_do = QLabel("Outer Diameter (mm):")
        rt_layout.addWidget(self.lbl_rt_do, 0, 0)
        self.rt_do = QDoubleSpinBox()
        self.rt_do.setRange(1.0, 2000.0)
        self.rt_do.setSingleStep(1.0)
        self.rt_do.setValue(200.0)
        rt_layout.addWidget(self.rt_do, 0, 1)
        self.lbl_rt_t = QLabel("Wall Thickness (mm):")
        rt_layout.addWidget(self.lbl_rt_t, 1, 0)
        self.rt_t = QDoubleSpinBox()
        self.rt_t.setRange(0.1, 500.0)
        self.rt_t.setSingleStep(1.0)
        self.rt_t.setValue(20.0)
        rt_layout.addWidget(self.rt_t, 1, 1)
        self.shape_inputs_stacked.addWidget(rt_widget)
        
        # Shape 1: Square Tube
        st_widget = QWidget()
        st_layout = QGridLayout(st_widget)
        st_layout.setContentsMargins(0, 5, 0, 5)
        self.lbl_st_w = QLabel("Outer Width (mm):")
        st_layout.addWidget(self.lbl_st_w, 0, 0)
        self.st_w = QDoubleSpinBox()
        self.st_w.setRange(1.0, 2000.0)
        self.st_w.setSingleStep(1.0)
        self.st_w.setValue(200.0)
        st_layout.addWidget(self.st_w, 0, 1)
        self.lbl_st_t = QLabel("Wall Thickness (mm):")
        st_layout.addWidget(self.lbl_st_t, 1, 0)
        self.st_t = QDoubleSpinBox()
        self.st_t.setRange(0.1, 500.0)
        self.st_t.setSingleStep(1.0)
        self.st_t.setValue(20.0)
        st_layout.addWidget(self.st_t, 1, 1)
        self.shape_inputs_stacked.addWidget(st_widget)
        
        # Shape 2: Rectangular Tube
        ret_widget = QWidget()
        ret_layout = QGridLayout(ret_widget)
        ret_layout.setContentsMargins(0, 5, 0, 5)
        self.lbl_ret_w = QLabel("Outer Width (mm):")
        ret_layout.addWidget(self.lbl_ret_w, 0, 0)
        self.ret_w = QDoubleSpinBox()
        self.ret_w.setRange(1.0, 2000.0)
        self.ret_w.setSingleStep(1.0)
        self.ret_w.setValue(200.0)
        ret_layout.addWidget(self.ret_w, 0, 1)
        self.lbl_ret_h = QLabel("Outer Height (mm):")
        ret_layout.addWidget(self.lbl_ret_h, 1, 0)
        self.ret_h = QDoubleSpinBox()
        self.ret_h.setRange(1.0, 2000.0)
        self.ret_h.setSingleStep(1.0)
        self.ret_h.setValue(300.0)
        ret_layout.addWidget(self.ret_h, 1, 1)
        self.lbl_ret_t = QLabel("Wall Thickness (mm):")
        ret_layout.addWidget(self.lbl_ret_t, 2, 0)
        self.ret_t = QDoubleSpinBox()
        self.ret_t.setRange(0.1, 500.0)
        self.ret_t.setSingleStep(1.0)
        self.ret_t.setValue(20.0)
        ret_layout.addWidget(self.ret_t, 2, 1)
        self.shape_inputs_stacked.addWidget(ret_widget)
        
        # Shape 3: I-Beam
        ib_widget = QWidget()
        ib_layout = QGridLayout(ib_widget)
        ib_layout.setContentsMargins(0, 5, 0, 5)
        self.lbl_ib_h = QLabel("Total Height (mm):")
        ib_layout.addWidget(self.lbl_ib_h, 0, 0)
        self.ib_h = QDoubleSpinBox()
        self.ib_h.setRange(1.0, 2000.0)
        self.ib_h.setSingleStep(1.0)
        self.ib_h.setValue(300.0)
        ib_layout.addWidget(self.ib_h, 0, 1)
        self.lbl_ib_w = QLabel("Flange Width (mm):")
        ib_layout.addWidget(self.lbl_ib_w, 1, 0)
        self.ib_w = QDoubleSpinBox()
        self.ib_w.setRange(1.0, 2000.0)
        self.ib_w.setSingleStep(1.0)
        self.ib_w.setValue(200.0)
        ib_layout.addWidget(self.ib_w, 1, 1)
        self.lbl_ib_tf = QLabel("Flange Thickness (mm):")
        ib_layout.addWidget(self.lbl_ib_tf, 2, 0)
        self.ib_tf = QDoubleSpinBox()
        self.ib_tf.setRange(0.1, 500.0)
        self.ib_tf.setSingleStep(1.0)
        self.ib_tf.setValue(20.0)
        ib_layout.addWidget(self.ib_tf, 2, 1)
        self.lbl_ib_tw = QLabel("Web Thickness (mm):")
        ib_layout.addWidget(self.lbl_ib_tw, 3, 0)
        self.ib_tw = QDoubleSpinBox()
        self.ib_tw.setRange(0.1, 500.0)
        self.ib_tw.setSingleStep(1.0)
        self.ib_tw.setValue(10.0)
        ib_layout.addWidget(self.ib_tw, 3, 1)
        self.shape_inputs_stacked.addWidget(ib_widget)
        
        # Shape 4: Round Bar (Solid)
        rb_widget = QWidget()
        rb_layout = QGridLayout(rb_widget)
        rb_layout.setContentsMargins(0, 5, 0, 5)
        self.lbl_rb_d = QLabel("Diameter (mm):")
        rb_layout.addWidget(self.lbl_rb_d, 0, 0)
        self.rb_d = QDoubleSpinBox()
        self.rb_d.setRange(1.0, 2000.0)
        self.rb_d.setSingleStep(1.0)
        self.rb_d.setValue(100.0)
        rb_layout.addWidget(self.rb_d, 0, 1)
        self.shape_inputs_stacked.addWidget(rb_widget)
        
        # Shape 5: Rectangular Bar (Solid)
        rebar_widget = QWidget()
        rebar_layout = QGridLayout(rebar_widget)
        rebar_layout.setContentsMargins(0, 5, 0, 5)
        self.lbl_rebar_w = QLabel("Width (mm):")
        rebar_layout.addWidget(self.lbl_rebar_w, 0, 0)
        self.rebar_w = QDoubleSpinBox()
        self.rebar_w.setRange(1.0, 2000.0)
        self.rebar_w.setSingleStep(1.0)
        self.rebar_w.setValue(100.0)
        rebar_layout.addWidget(self.rebar_w, 0, 1)
        self.lbl_rebar_h = QLabel("Height (mm):")
        rebar_layout.addWidget(self.lbl_rebar_h, 1, 0)
        self.rebar_h = QDoubleSpinBox()
        self.rebar_h.setRange(1.0, 2000.0)
        self.rebar_h.setSingleStep(1.0)
        self.rebar_h.setValue(200.0)
        rebar_layout.addWidget(self.rebar_h, 1, 1)
        self.shape_inputs_stacked.addWidget(rebar_widget)
        
        # Shape 6: General (Custom)
        gen_widget = QWidget()
        gen_layout = QGridLayout(gen_widget)
        gen_layout.setContentsMargins(0, 5, 0, 5)
        self.lbl_gen_a = QLabel("Area (A) [mm²]:")
        gen_layout.addWidget(self.lbl_gen_a, 0, 0)
        self.gen_a = QDoubleSpinBox()
        self.gen_a.setRange(1.0, 1e8)
        self.gen_a.setValue(11300.0)
        gen_layout.addWidget(self.gen_a, 0, 1)
        self.lbl_gen_iy = QLabel("Iy [mm⁴]:")
        gen_layout.addWidget(self.lbl_gen_iy, 1, 0)
        self.gen_iy = QDoubleSpinBox()
        self.gen_iy.setRange(1.0, 1e12)
        self.gen_iy.setValue(4.64e7)
        gen_layout.addWidget(self.gen_iy, 1, 1)
        self.lbl_gen_iz = QLabel("Iz [mm⁴]:")
        gen_layout.addWidget(self.lbl_gen_iz, 2, 0)
        self.gen_iz = QDoubleSpinBox()
        self.gen_iz.setRange(1.0, 1e12)
        self.gen_iz.setValue(4.64e7)
        gen_layout.addWidget(self.gen_iz, 2, 1)
        self.lbl_gen_j = QLabel("J [mm⁴]:")
        gen_layout.addWidget(self.lbl_gen_j, 3, 0)
        self.gen_j = QDoubleSpinBox()
        self.gen_j.setRange(1.0, 1e12)
        self.gen_j.setValue(9.28e7)
        gen_layout.addWidget(self.gen_j, 3, 1)
        self.lbl_gen_ymax = QLabel("y_max (fiber y) [mm]:")
        gen_layout.addWidget(self.lbl_gen_ymax, 4, 0)
        self.gen_ymax = QDoubleSpinBox()
        self.gen_ymax.setRange(0.1, 2000.0)
        self.gen_ymax.setValue(100.0)
        gen_layout.addWidget(self.gen_ymax, 4, 1)
        self.lbl_gen_zmax = QLabel("z_max (fiber z) [mm]:")
        gen_layout.addWidget(self.lbl_gen_zmax, 5, 0)
        self.gen_zmax = QDoubleSpinBox()
        self.gen_zmax.setRange(0.1, 2000.0)
        self.gen_zmax.setValue(100.0)
        gen_layout.addWidget(self.gen_zmax, 5, 1)
        self.shape_inputs_stacked.addWidget(gen_widget)
        
        sec_layout.addWidget(self.shape_inputs_stacked)
        
        # Live calculation readouts (Dynamic stats box in mm)
        self.lbl_sec_area = QLabel("Area (A): 1.13e+4 mm²")
        self.lbl_sec_area.setStyleSheet("color: #06b6d4; font-weight: bold; font-size: 11px;")
        sec_layout.addWidget(self.lbl_sec_area)
        
        self.lbl_sec_iy = QLabel("Iy: 4.64e+07 mm⁴")
        self.lbl_sec_iy.setStyleSheet("color: #06b6d4; font-weight: bold; font-size: 11px;")
        sec_layout.addWidget(self.lbl_sec_iy)
        
        self.lbl_sec_iz = QLabel("Iz: 4.64e+07 mm⁴")
        self.lbl_sec_iz.setStyleSheet("color: #06b6d4; font-weight: bold; font-size: 11px;")
        sec_layout.addWidget(self.lbl_sec_iz)
        
        self.lbl_sec_j = QLabel("Polar J: 9.28e+07 mm⁴")
        self.lbl_sec_j.setStyleSheet("color: #06b6d4; font-weight: bold; font-size: 11px;")
        sec_layout.addWidget(self.lbl_sec_j)
        
        layout.addWidget(self.sec_grp)
        
        # 2. Option A: Parameterized Straight Beam
        self.param_grp = QGroupBox("Option A: Parameterized Straight Beam")
        g_layout = QGridLayout(self.param_grp)
        
        self.lbl_sp_length = QLabel("Total Beam Length (mm):")
        g_layout.addWidget(self.lbl_sp_length, 0, 0)
        self.sp_length = QDoubleSpinBox()
        self.sp_length.setRange(100.0, 50000.0)
        self.sp_length.setSingleStep(100.0)
        self.sp_length.setValue(2000.0)
        self.sp_length.valueChanged.connect(self.update_visualization)
        g_layout.addWidget(self.sp_length, 0, 1)
        layout.addWidget(self.param_grp)
        
        # 3. Option B: STEP File Import
        self.step_grp = QGroupBox("Option B: Import STEP Wireframe")
        step_layout = QVBoxLayout(self.step_grp)
        
        self.btn_import_step = QPushButton("📂 Import STEP File (.stp)")
        self.btn_import_step.setStyleSheet("background-color: #1e2942; border: 1px solid #06b6d4; padding: 10px; font-weight: bold;")
        self.btn_import_step.clicked.connect(self.import_step_file)
        step_layout.addWidget(self.btn_import_step)
        
        self.lbl_step_status = QLabel("No STEP file loaded. Using Parameterized Beam.")
        self.lbl_step_status.setWordWrap(True)
        self.lbl_step_status.setStyleSheet("color: #9ca3af; font-size: 12px;")
        step_layout.addWidget(self.lbl_step_status)
        
        self.btn_clear_step = QPushButton("🔄 Reset to Parametric Beam")
        self.btn_clear_step.setStyleSheet("background-color: #1e2942; border: 1px solid #ef4444; padding: 5px; color: #ef4444;")
        self.btn_clear_step.clicked.connect(self.clear_step_import)
        self.btn_clear_step.setVisible(False)
        step_layout.addWidget(self.btn_clear_step)
        
        layout.addWidget(self.step_grp)
        layout.addStretch()
        
        self.stacked_widget.addWidget(page)
        
        # Setup shape connections
        self.combo_shape.currentIndexChanged.connect(self.shape_inputs_stacked.setCurrentIndex)
        self.combo_shape.currentIndexChanged.connect(self.update_section_calculation)
        self.combo_shape.currentIndexChanged.connect(self.update_visualization)
        
        # Link all sub-widgets valueChanged signals to update calculations
        for spin in [
            self.rt_do, self.rt_t, self.st_w, self.st_t, self.ret_w, self.ret_h, self.ret_t,
            self.ib_h, self.ib_w, self.ib_tf, self.ib_tw, self.rb_d, self.rebar_w, self.rebar_h,
            self.gen_a, self.gen_iy, self.gen_iz, self.gen_j, self.gen_ymax, self.gen_zmax
        ]:
            spin.valueChanged.connect(self.update_section_calculation)
            spin.valueChanged.connect(self.update_visualization)

    def import_step_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open STEP File", "", "STEP Files (*.stp *.step)"
        )
        if not file_path:
            return
            
        parser = STEPParser()
        try:
            result = parser.parse(file_path)
        except Exception as e:
            self.lbl_step_status.setText(f"❌ Failed to parse STEP file: {e}")
            return
            
        # Scan header content to auto-detect CAD model unit scale
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.read(50000)
        except Exception:
            header = ""
            
        cad_unit = "mm" # default fallback
        if ".METRE." in header and ".MILLI." not in header:
            cad_unit = "m"
        elif "INCH" in header:
            cad_unit = "in"
        elif ".CENTI." in header:
            cad_unit = "cm"
            
        # Determine scale factor from CAD native unit to current UI unit system
        scale = 1.0
        is_metric = (self.unit_system == "Metric")
        if is_metric:
            if cad_unit == "m":
                scale = 1000.0
            elif cad_unit == "cm":
                scale = 10.0
            elif cad_unit == "in":
                scale = 25.4
        else:
            if cad_unit == "m":
                scale = 1000.0 / 25.4
            elif cad_unit == "cm":
                scale = 10.0 / 25.4
            elif cad_unit == "mm":
                scale = 1.0 / 25.4
                
        # Scale the coordinates of the nodes
        for node in result['nodes'].values():
            node.point.x *= scale
            node.point.y *= scale
            node.point.z *= scale
            
        self.model.load_geometry(result)
        
        self.lbl_step_status.setText(
            f"✅ Loaded: {file_path.split('/')[-1]}\n"
            f"• CAD Unit: {cad_unit.upper()} (Auto-Scaled to {self.unit_system})\n"
            f"• Coords: {len(self.model.geometry_nodes)} unique vertices\n"
            f"• Edges: {len(self.model.geometry_edges)} parsed segments"
        )
        
        self.btn_clear_step.setVisible(True)
        self.param_grp.setEnabled(False)  # Disable parameterized inputs
        self.update_visualization()
        
    def clear_step_import(self):
        self.model.geometry_nodes.clear()
        self.model.geometry_edges.clear()
        self.model.geometry_faces.clear()
        self.model.source_file = ""
        self.lbl_step_status.setText("No STEP file loaded. Using Parameterized Beam.")
        self.btn_clear_step.setVisible(False)
        self.param_grp.setEnabled(True)
        self.create_parametric_geometry()
        self.update_visualization()

    def recalculate_section_properties(self):
        shape = self.combo_shape.currentText()
        is_metric = (self.unit_system == "Metric")
        
        if is_metric:
            A = 11300.0
            Iy = 4.64e7
            Iz = 4.64e7
            J = 9.28e7
            y_max = 100.0
            z_max = 100.0
        else:
            A = 11300.0 * (MM_TO_IN**2)
            Iy = 4.64e7 * (MM_TO_IN**4)
            Iz = 4.64e7 * (MM_TO_IN**4)
            J = 9.28e7 * (MM_TO_IN**4)
            y_max = 100.0 * MM_TO_IN
            z_max = 100.0 * MM_TO_IN
        
        if shape == "Round Tube":
            do = self.rt_do.value()
            t = self.rt_t.value()
            if t >= do/2:
                t = do/2 - 0.1
                self.rt_t.blockSignals(True)
                self.rt_t.setValue(t)
                self.rt_t.blockSignals(False)
            ro = do / 2.0
            ri = ro - t
            A = np.pi * (ro**2 - ri**2)
            Iy = Iz = (np.pi / 4.0) * (ro**4 - ri**4)
            J = 2.0 * Iy
            y_max = z_max = ro
            
        elif shape == "Square Tube":
            W = self.st_w.value()
            t = self.st_t.value()
            if t >= W/2:
                t = W/2 - 0.1
                self.st_t.blockSignals(True)
                self.st_t.setValue(t)
                self.st_t.blockSignals(False)
            w = W - 2.0*t
            A = W**2 - w**2
            Iy = Iz = (1.0 / 12.0) * (W**4 - w**4)
            J = (W - t)**3 * t
            y_max = z_max = W / 2.0
            
        elif shape == "Rectangular Tube":
            W = self.ret_w.value()
            H = self.ret_h.value()
            t = self.ret_t.value()
            min_dim = min(W, H)
            if t >= min_dim/2:
                t = min_dim/2 - 0.1
                self.ret_t.blockSignals(True)
                self.ret_t.setValue(t)
                self.ret_t.blockSignals(False)
            w = W - 2.0*t
            h = H - 2.0*t
            A = W*H - w*h
            Iy = (1.0 / 12.0) * (W * H**3 - w * h**3)
            Iz = (1.0 / 12.0) * (H * W**3 - h * w**3)
            J = (2.0 * t * (W - t)**2 * (H - t)**2) / ((W - t) + (H - t))
            y_max = W / 2.0
            z_max = H / 2.0
            
        elif shape == "I-Beam":
            H = self.ib_h.value()
            W = self.ib_w.value()
            tf = self.ib_tf.value()
            tw = self.ib_tw.value()
            if tf >= H/2:
                tf = H/2 - 0.1
                self.ib_tf.blockSignals(True)
                self.ib_tf.setValue(tf)
                self.ib_tf.blockSignals(False)
            if tw >= W:
                tw = W - 0.1
                self.ib_tw.blockSignals(True)
                self.ib_tw.setValue(tw)
                self.ib_tw.blockSignals(False)
            A = 2.0 * (W * tf) + tw * (H - 2.0 * tf)
            Iy = (1.0 / 12.0) * W * H**3 - (1.0 / 12.0) * (W - tw) * (H - 2.0 * tf)**3
            Iz = 2.0 * ((1.0 / 12.0) * tf * W**3) + (1.0 / 12.0) * (H - 2.0 * tf) * tw**3
            J = (1.0 / 3.0) * (2.0 * W * tf**3 + (H - 2.0 * tf) * tw**3)
            y_max = W / 2.0
            z_max = H / 2.0
            
        elif shape == "Round Bar (Solid)":
            d = self.rb_d.value()
            ro = d / 2.0
            A = np.pi * ro**2
            Iy = Iz = (np.pi / 4.0) * ro**4
            J = 2.0 * Iy
            y_max = z_max = ro
            
        elif shape == "Rectangular Bar (Solid)":
            W = self.rebar_w.value()
            H = self.rebar_h.value()
            A = W * H
            Iy = (1.0 / 12.0) * W * H**3
            Iz = (1.0 / 12.0) * H * W**3
            a = max(W, H)
            b = min(W, H)
            J = a * b**3 * (1.0/3.0 - 0.21 * (b/a) * (1.0 - b**4 / (12.0 * a**4)))
            y_max = W / 2.0
            z_max = H / 2.0
            
        elif shape == "General (Custom)":
            A = self.gen_a.value()
            Iy = self.gen_iy.value()
            Iz = self.gen_iz.value()
            J = self.gen_j.value()
            y_max = self.gen_ymax.value()
            z_max = self.gen_zmax.value()
            
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.y_max = y_max
        self.z_max = z_max
        
    def update_section_calculation(self):
        self.recalculate_section_properties()
        is_metric = (self.unit_system == "Metric")
        u_len2 = "mm²" if is_metric else "in²"
        u_len4 = "mm⁴" if is_metric else "in⁴"
        
        self.lbl_sec_area.setText(f"Area (A): {self.A:.2e} {u_len2}")
        self.lbl_sec_iy.setText(f"Iy: {self.Iy:.2e} {u_len4}")
        self.lbl_sec_iz.setText(f"Iz: {self.Iz:.2e} {u_len4}")
        self.lbl_sec_j.setText(f"Polar J: {self.J:.2e} {u_len4}")

    def create_parametric_geometry(self):
        L = self.sp_length.value()
        n0 = GeometryNode(id=0, label="Node 0", point=Point3D(0.0, 0.0, 0.0))
        n1 = GeometryNode(id=1, label="Node 1", point=Point3D(L, 0.0, 0.0))
        e0 = GeometryEdge(id=0, label="Edge 0", start_node_id=0, end_node_id=1)
        
        geom_data = {
            'nodes': {0: n0, 1: n1},
            'edges': {0: e0},
            'faces': {},
            'source_file': 'Parametric Beam'
        }
        self.model.load_geometry(geom_data)

    def get_preview_mesh(self):
        if self.model.source_file == 'Parametric Beam' or not self.model.geometry_nodes:
            self.create_parametric_geometry()
            
        mat_name = self.combo_mat.currentText()
        self.recalculate_section_properties()
        
        mat = MaterialDef(
            name=mat_name,
            youngs_modulus=self.sp_E.value(),
            poissons_ratio=self.sp_nu.value(),
            yield_strength=self.sp_yield.value()
        )
        self.model.materials[mat.name] = mat
        
        shape = self.combo_shape.currentText()
        section = SectionDef(
            name=shape,
            area=self.A,
            iy=self.Iy,
            iz=self.Iz,
            j=self.J,
            outer_radius=max(self.y_max, self.z_max),
            inner_radius=self.y_max * 0.8 if "Tube" in shape or "Beam" in shape else 0.0
        )
        self.model.sections[section.name] = section
        
        for eid in self.model.geometry_edges:
            self.model.edge_assignments[eid] = PropertyAssignment(
                material_name=mat.name,
                section_name=section.name
            )
            
        mesher = BeamMesher(n_divisions=self.sp_elements.value())
        try:
            mesher.mesh(self.model)
        except Exception as e:
            print(f"Error in meshing: {e}")

    def parse_node_ids(self, text):
        text = text.strip().lower()
        if not text:
            return []
            
        if self.model.source_file == 'Parametric Beam' or not self.model.geometry_nodes:
            self.create_parametric_geometry()
            
        valid_ids = list(self.model.geometry_nodes.keys())
        
        if text == "all":
            return valid_ids
            
        nodes = []
        parts = text.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                subparts = part.split('-')
                if len(subparts) == 2:
                    try:
                        start = int(subparts[0].strip())
                        end = int(subparts[1].strip())
                        for idx in range(start, end + 1):
                            if idx in valid_ids:
                                nodes.append(idx)
                    except ValueError:
                        pass
            else:
                try:
                    idx = int(part)
                    if idx in valid_ids:
                        nodes.append(idx)
                except ValueError:
                    pass
        return sorted(list(set(nodes)))

    def get_current_boundary_conditions(self):
        supports = []
        loads = []
        
        from wnfea.boundary.conditions import SupportDef, LoadDef, DOFConstraint, DOFType
        
        # 1. Parse supports
        for r in range(self.table_supports.rowCount()):
            node_id_item = self.table_supports.item(r, 0)
            if node_id_item is None:
                continue
            node_ids = self.parse_node_ids(node_id_item.text())
            
            ux_val = DOFType.FIXED if self.table_supports.item(r, 1).checkState() == Qt.Checked else DOFType.FREE
            uy_val = DOFType.FIXED if self.table_supports.item(r, 2).checkState() == Qt.Checked else DOFType.FREE
            uz_val = DOFType.FIXED if self.table_supports.item(r, 3).checkState() == Qt.Checked else DOFType.FREE
            rx_val = DOFType.FIXED if self.table_supports.item(r, 4).checkState() == Qt.Checked else DOFType.FREE
            ry_val = DOFType.FIXED if self.table_supports.item(r, 5).checkState() == Qt.Checked else DOFType.FREE
            rz_val = DOFType.FIXED if self.table_supports.item(r, 6).checkState() == Qt.Checked else DOFType.FREE
            
            for nid in node_ids:
                sup = SupportDef(
                    node_id=nid,
                    is_geometry_node=True,
                    ux=DOFConstraint(ux_val),
                    uy=DOFConstraint(uy_val),
                    uz=DOFConstraint(uz_val),
                    rx=DOFConstraint(rx_val),
                    ry=DOFConstraint(ry_val),
                    rz=DOFConstraint(rz_val)
                )
                supports.append(sup)
                
        # 2. Parse loads
        for r in range(self.table_loads.rowCount()):
            node_id_item = self.table_loads.item(r, 0)
            if node_id_item is None:
                continue
            node_ids = self.parse_node_ids(node_id_item.text())
            
            try:
                fx = float(self.table_loads.item(r, 1).text())
            except (ValueError, AttributeError, TypeError):
                fx = 0.0
            try:
                fy = float(self.table_loads.item(r, 2).text())
            except (ValueError, AttributeError, TypeError):
                fy = 0.0
            try:
                fz = float(self.table_loads.item(r, 3).text())
            except (ValueError, AttributeError, TypeError):
                fz = 0.0
            try:
                mx = float(self.table_loads.item(r, 4).text())
            except (ValueError, AttributeError, TypeError):
                mx = 0.0
            try:
                my = float(self.table_loads.item(r, 5).text())
            except (ValueError, AttributeError, TypeError):
                my = 0.0
            try:
                mz = float(self.table_loads.item(r, 6).text())
            except (ValueError, AttributeError, TypeError):
                mz = 0.0
                
            for nid in node_ids:
                load = LoadDef(
                    node_id=nid,
                    is_geometry_node=True,
                    fx=fx, fy=fy, fz=fz,
                    mx=mx, my=my, mz=mz
                )
                loads.append(load)
                
        return supports, loads

    def add_support_row(self):
        self.table_supports.blockSignals(True)
        row = self.table_supports.rowCount()
        self.table_supports.insertRow(row)
        
        # Node ID
        item_node = QTableWidgetItem("0")
        item_node.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
        self.table_supports.setItem(row, 0, item_node)
        
        # Checkboxes for 6 DOFs
        for col in range(1, 7):
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setCheckState(Qt.Unchecked)
            self.table_supports.setItem(row, col, item)
            
        self.table_supports.blockSignals(False)
        self.update_visualization()
        
    def remove_support_row(self):
        selected = self.table_supports.currentRow()
        if selected >= 0:
            self.table_supports.removeRow(selected)
            self.update_visualization()
            
    def add_load_row(self):
        self.table_loads.blockSignals(True)
        row = self.table_loads.rowCount()
        self.table_loads.insertRow(row)
        
        # Node ID
        item_node = QTableWidgetItem("1")
        item_node.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
        self.table_loads.setItem(row, 0, item_node)
        
        # Forces/Moments (col 1-6)
        for col in range(1, 7):
            item = QTableWidgetItem("0.0")
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
            self.table_loads.setItem(row, col, item)
            
        self.table_loads.blockSignals(False)
        self.update_visualization()
        
    def remove_load_row(self):
        selected = self.table_loads.currentRow()
        if selected >= 0:
            self.table_loads.removeRow(selected)
            self.update_visualization()
            
    def refresh_setup_nodes(self):
        pass

    def update_visualization(self):
        self.geom_canvas.ax.clear()
        
        # Remove old colorbar if it exists
        if hasattr(self, 'colorbar') and self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None
            
        self.recalculate_section_properties()
        
        idx = self.stacked_widget.currentIndex()
        
        # Build the preview mesh
        self.get_preview_mesh()
        
        nodes = self.model.mesh_nodes
        if nodes is None or len(nodes) == 0:
            return
            
        min_bounds = nodes.min(axis=0)
        max_bounds = nodes.max(axis=0)
        centers = (min_bounds + max_bounds) / 2.0
        ranges = (max_bounds - min_bounds) / 2.0
        max_range = max(ranges.max(), 0.5)
        
        self.geom_canvas.ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
        self.geom_canvas.ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
        self.geom_canvas.ax.set_zlim(centers[2] - max_range, centers[2] + max_range)
        
        u_len = "mm" if self.unit_system == "Metric" else "in"
        self.geom_canvas.ax.set_xlabel(f'X ({u_len})')
        self.geom_canvas.ax.set_ylabel(f'Y ({u_len})')
        self.geom_canvas.ax.set_zlabel(f'Z ({u_len})')
        self.geom_canvas.ax.set_box_aspect((1, 1, 1))
        
        ro = max(self.y_max, self.z_max)
        shape = self.combo_shape.currentText()
        ri = ro * 0.8 if "Tube" in shape or "Beam" in shape else 0.0
        res = 12
        
        if idx == 4 and self.model.displacements is not None:
            # Page 4 (Results Page): Render deformed shape colored by stress
            sf = self.sp_scale.value()
            defs = self.model.displacements.reshape(-1, 6)
            deformed_nodes = nodes + defs[:, :3] * sf
            
            all_stresses = []
            for elem_idx in range(len(self.model.mesh_elements)):
                res_elem = self.model.element_results.get(elem_idx, {}) if self.model.element_results else {}
                stress_val = res_elem.get('von_mises', {}).get('max', 0.0)
                all_stresses.append(stress_val)
                
            vmin, vmax = min(all_stresses) if all_stresses else 0.0, max(all_stresses) if all_stresses else 0.0
            if vmax - vmin < 1e-3:
                vmin -= 1e-3
                vmax += 1e-3
            norm = plt.Normalize(vmin, vmax)
            cmap = plt.cm.jet
            
            for elem_idx, (n1, n2) in enumerate(self.model.mesh_elements):
                pt1, pt2 = deformed_nodes[n1], deformed_nodes[n2]
                c = cmap(norm(all_stresses[elem_idx]))
                vec = pt2 - pt1
                mag = np.linalg.norm(vec)
                if mag < 1e-9: continue
                vn = vec / mag
                
                v1 = np.array([1, 0, 0]) if np.isclose(np.abs(vn[2]), 1) else np.array([0, 0, 1])
                v1 = v1 - np.dot(v1, vn)*vn
                v1 /= np.linalg.norm(v1)
                v2 = np.cross(vn, v1)
                
                th = np.linspace(0, 2*np.pi, res)
                cout = ro * (np.outer(np.cos(th), v1) + np.outer(np.sin(th), v2))
                
                v_out = []
                for j in range(res - 1):
                    v_out.append([pt1+cout[j], pt1+cout[j+1], pt2+cout[j+1], pt2+cout[j]])
                v_out.append([pt1+cout[res-1], pt1+cout[0], pt2+cout[0], pt2+cout[res-1]])
                
                self.geom_canvas.ax.add_collection3d(Poly3DCollection(v_out, facecolor=c, edgecolor='k', lw=0.15, alpha=0.9))
                
                if ri > 0:
                    cin = ri * (np.outer(np.cos(th), v1) + np.outer(np.sin(th), v2))
                    v_in = []
                    for j in range(res - 1):
                        v_in.append([pt1+cin[j], pt1+cin[j+1], pt2+cin[j+1], pt2+cin[j]])
                    v_in.append([pt1+cin[res-1], pt1+cin[0], pt2+cin[0], pt2+cin[res-1]])
                    self.geom_canvas.ax.add_collection3d(Poly3DCollection(v_in, facecolor='gray', edgecolor='k', lw=0.1, alpha=0.4))
                    
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            u_stress = "MPa" if self.unit_system == "Metric" else "psi"
            self.colorbar = self.geom_canvas.fig.colorbar(sm, ax=self.geom_canvas.ax, shrink=0.6, aspect=15, pad=0.05)
            self.colorbar.set_label(f"Von Mises Stress ({u_stress})", color='#9ca3af')
            self.colorbar.ax.yaxis.set_tick_params(color='#9ca3af')
            plt.setp(self.colorbar.ax.get_yticklabels(), color='#9ca3af')
            
            self.geom_canvas.ax.scatter(deformed_nodes[:,0], deformed_nodes[:,1], deformed_nodes[:,2], color='white', s=50)
            
        else:
            # Undeformed shape
            for n1, n2 in self.model.mesh_elements:
                pt1, pt2 = nodes[n1], nodes[n2]
                vec = pt2 - pt1
                mag = np.linalg.norm(vec)
                if mag < 1e-9: continue
                vn = vec / mag
                
                v1 = np.array([1, 0, 0]) if np.isclose(np.abs(vn[2]), 1) else np.array([0, 0, 1])
                v1 = v1 - np.dot(v1, vn)*vn
                v1 /= np.linalg.norm(v1)
                v2 = np.cross(vn, v1)
                
                th = np.linspace(0, 2*np.pi, res)
                cout = ro * (np.outer(np.cos(th), v1) + np.outer(np.sin(th), v2))
                
                v_out = []
                for j in range(res - 1):
                    v_out.append([pt1+cout[j], pt1+cout[j+1], pt2+cout[j+1], pt2+cout[j]])
                v_out.append([pt1+cout[res-1], pt1+cout[0], pt2+cout[0], pt2+cout[res-1]])
                
                self.geom_canvas.ax.add_collection3d(Poly3DCollection(v_out, facecolor='#06b6d4', edgecolor='k', lw=0.15, alpha=0.8))
                
                if ri > 0:
                    cin = ri * (np.outer(np.cos(th), v1) + np.outer(np.sin(th), v2))
                    v_in = []
                    for j in range(res - 1):
                        v_in.append([pt1+cin[j], pt1+cin[j+1], pt2+cin[j+1], pt2+cin[j]])
                    v_in.append([pt1+cin[res-1], pt1+cin[0], pt2+cin[0], pt2+cin[res-1]])
                    self.geom_canvas.ax.add_collection3d(Poly3DCollection(v_in, facecolor='gray', edgecolor='k', lw=0.1, alpha=0.3))
                    
            if idx == 1:
                # Highlight mesh nodes in yellow
                self.geom_canvas.ax.scatter(nodes[:,0], nodes[:,1], nodes[:,2], color='#eab308', s=60, zorder=5)
            else:
                self.geom_canvas.ax.scatter(nodes[:,0], nodes[:,1], nodes[:,2], color='white', s=50, zorder=5)
                
            if idx in [2, 3]:
                # Draw boundary conditions
                supports, loads = self.get_current_boundary_conditions()
                geom_nodes = self.model.geometry_nodes
                
                for sup in supports:
                    if sup.node_id in geom_nodes:
                        coord = geom_nodes[sup.node_id].point.to_array()
                        self.geom_canvas.ax.scatter([coord[0]], [coord[1]], [coord[2]], color='#22c55e', marker='^', s=120, zorder=10)
                        
                for load in loads:
                    if load.node_id in geom_nodes:
                        coord = geom_nodes[load.node_id].point.to_array()
                        fx, fy, fz = load.fx, load.fy, load.fz
                        f_mag = np.linalg.norm([fx, fy, fz])
                        if f_mag > 1e-6:
                            dx, dy, dz = fx/f_mag, fy/f_mag, fz/f_mag
                            L_arrow = 0.3 * max_range
                            self.geom_canvas.ax.quiver(
                                coord[0], coord[1], coord[2], dx, dy, dz,
                                length=L_arrow, color='#ef4444', lw=2, arrow_length_ratio=0.3, zorder=15
                            )
                            
        self.geom_canvas.draw()

    def create_meshing_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 40)
        
        header = QLabel("💠 Finite Element Meshing")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        layout.addWidget(header)
        
        grp = QGroupBox("Discretization")
        g_layout = QGridLayout(grp)
        
        g_layout.addWidget(QLabel("Number of Elements:"), 0, 0)
        self.sp_elements = QSpinBox()
        self.sp_elements.setRange(1, 20)
        self.sp_elements.setValue(2)
        g_layout.addWidget(self.sp_elements, 0, 1)
        
        layout.addWidget(grp)
        layout.addStretch()
        self.stacked_widget.addWidget(page)

    def create_setup_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(15, 15, 15, 15)
        
        header = QLabel("⚡ Problem Setup")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        layout.addWidget(header)
        
        setup_tabs = QTabWidget()
        
        # --- Tab 1: Materials ---
        mat_tab = QWidget()
        mat_layout = QVBoxLayout(mat_tab)
        mat_layout.setContentsMargins(10, 10, 10, 10)
        
        m_grp = QGroupBox("Material Properties")
        m_grid = QGridLayout(m_grp)
        
        self.combo_mat = QComboBox()
        self.combo_mat.addItems(["Structural Steel", "Aluminum 6061-T6", "Titanium Grade 5", "Custom"])
        m_grid.addWidget(QLabel("Preset:"), 0, 0)
        m_grid.addWidget(self.combo_mat, 0, 1)
        
        self.lbl_E_name = QLabel("Young's Modulus (MPa):")
        m_grid.addWidget(self.lbl_E_name, 1, 0)
        self.sp_E = QDoubleSpinBox()
        self.sp_E.setRange(100.0, 1000000.0)
        self.sp_E.setValue(200000.0)
        m_grid.addWidget(self.sp_E, 1, 1)
        
        m_grid.addWidget(QLabel("Poisson's Ratio:"), 2, 0)
        self.sp_nu = QDoubleSpinBox()
        self.sp_nu.setRange(0.0, 0.49)
        self.sp_nu.setSingleStep(0.01)
        self.sp_nu.setValue(0.27)
        m_grid.addWidget(self.sp_nu, 2, 1)
        
        self.lbl_yield_name = QLabel("Yield Strength (MPa):")
        m_grid.addWidget(self.lbl_yield_name, 3, 0)
        self.sp_yield = QDoubleSpinBox()
        self.sp_yield.setRange(1.0, 10000.0)
        self.sp_yield.setValue(250.0)
        m_grid.addWidget(self.sp_yield, 3, 1)
        
        self.lbl_density_name = QLabel("Density (tonne/mm³):")
        m_grid.addWidget(self.lbl_density_name, 4, 0)
        self.sp_density = QDoubleSpinBox()
        self.sp_density.setDecimals(12)
        self.sp_density.setRange(1e-12, 1.0)
        self.sp_density.setValue(7.85e-9)
        self.sp_density.setSingleStep(1e-9)
        m_grid.addWidget(self.sp_density, 4, 1)
        
        mat_layout.addWidget(m_grp)
        mat_layout.addStretch()
        setup_tabs.addTab(mat_tab, "Materials")
        
        # --- Tab 2: Boundary Conditions (Supports) ---
        sup_tab = QWidget()
        sup_layout = QVBoxLayout(sup_tab)
        sup_layout.setContentsMargins(10, 10, 10, 10)
        
        self.table_supports = QTableWidget()
        self.table_supports.setColumnCount(7)
        self.table_supports.setHorizontalHeaderLabels(["Node ID(s)", "Ux", "Uy", "Uz", "Rx", "Ry", "Rz"])
        self.table_supports.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        sup_layout.addWidget(self.table_supports)
        
        sup_btns = QHBoxLayout()
        self.btn_add_sup = QPushButton("➕ Add Support")
        self.btn_add_sup.clicked.connect(self.add_support_row)
        self.btn_rem_sup = QPushButton("❌ Remove Selected")
        self.btn_rem_sup.clicked.connect(self.remove_support_row)
        sup_btns.addWidget(self.btn_add_sup)
        sup_btns.addWidget(self.btn_rem_sup)
        sup_layout.addLayout(sup_btns)
        
        setup_tabs.addTab(sup_tab, "Supports (Fixed DOFs)")
        
        # --- Tab 3: Applied Loads ---
        load_tab = QWidget()
        load_layout = QVBoxLayout(load_tab)
        load_layout.setContentsMargins(10, 10, 10, 10)
        
        self.table_loads = QTableWidget()
        self.table_loads.setColumnCount(7)
        self.table_loads.setHorizontalHeaderLabels(["Node ID(s)", "Fx (N)", "Fy (N)", "Fz (N)", "Mx (N-mm)", "My (N-mm)", "Mz (N-mm)"])
        self.table_loads.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        load_layout.addWidget(self.table_loads)
        
        load_btns = QHBoxLayout()
        self.btn_add_load = QPushButton("➕ Add Load")
        self.btn_add_load.clicked.connect(self.add_load_row)
        self.btn_rem_load = QPushButton("❌ Remove Selected")
        self.btn_rem_load.clicked.connect(self.remove_load_row)
        load_btns.addWidget(self.btn_add_load)
        load_btns.addWidget(self.btn_rem_load)
        load_layout.addLayout(load_btns)
        
        setup_tabs.addTab(load_tab, "Loads")
        
        layout.addWidget(setup_tabs)
        self.stacked_widget.addWidget(page)
        
        # Default row initialization
        self.add_support_row()
        item_node = self.table_supports.item(0, 0)
        if item_node is not None:
            item_node.setText("0")
        for col in range(1, 7):
            self.table_supports.item(0, col).setCheckState(Qt.Checked)
            
        self.add_load_row()
        item_node_load = self.table_loads.item(0, 0)
        if item_node_load is not None:
            item_node_load.setText("1")
        item_fy = self.table_loads.item(0, 2)
        if item_fy is not None:
            item_fy.setText("-1000.0")

    def create_solving_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 40)
        
        header = QLabel("⚙️ Solving Execution")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        layout.addWidget(header)
        
        self.run_btn = QPushButton("🚀 RUN SIMULATION")
        self.run_btn.setObjectName("RunBtn")
        self.run_btn.clicked.connect(self.run_simulation)
        layout.addWidget(self.run_btn)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(QLabel("Solver Output Log:"))
        layout.addWidget(self.log_output)
        
        self.stacked_widget.addWidget(page)

    def create_results_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(15, 15, 15, 15)
        
        metrics_layout = QHBoxLayout()
        
        grp1 = QWidget()
        grp1.setStyleSheet("background-color: #1e2942; border-radius: 8px;")
        l1 = QVBoxLayout(grp1)
        self.lbl_metric_deflection_title = QLabel("Total Tip Deflection (mm)")
        self.lbl_metric_deflection_title.setObjectName("MetricTitle")
        self.lbl_metric_deflection_title.setAlignment(Qt.AlignCenter)
        self.lbl_deflection = QLabel("0.0")
        self.lbl_deflection.setObjectName("MetricValue")
        self.lbl_deflection.setAlignment(Qt.AlignCenter)
        l1.addWidget(self.lbl_metric_deflection_title)
        l1.addWidget(self.lbl_deflection)
        metrics_layout.addWidget(grp1)
        
        grp2 = QWidget()
        grp2.setStyleSheet("background-color: #1e2942; border-radius: 8px;")
        l2 = QVBoxLayout(grp2)
        self.lbl_metric_stress_title = QLabel("Max Von Mises (MPa)")
        self.lbl_metric_stress_title.setObjectName("MetricTitle")
        self.lbl_metric_stress_title.setAlignment(Qt.AlignCenter)
        self.lbl_stress = QLabel("0.0")
        self.lbl_stress.setObjectName("MetricValue")
        self.lbl_stress.setAlignment(Qt.AlignCenter)
        l2.addWidget(self.lbl_metric_stress_title)
        l2.addWidget(self.lbl_stress)
        metrics_layout.addWidget(grp2)
        
        grp3 = QWidget()
        grp3.setStyleSheet("background-color: #1e2942; border-radius: 8px;")
        l3 = QVBoxLayout(grp3)
        lbl_safety_title = QLabel("Safety Factor")
        lbl_safety_title.setObjectName("MetricTitle")
        lbl_safety_title.setAlignment(Qt.AlignCenter)
        self.lbl_safety = QLabel("0.0")
        self.lbl_safety.setObjectName("MetricValue")
        self.lbl_safety.setAlignment(Qt.AlignCenter)
        l3.addWidget(lbl_safety_title)
        l3.addWidget(self.lbl_safety)
        metrics_layout.addWidget(grp3)
        
        layout.addLayout(metrics_layout)
        
        vis_grp = QGroupBox("Visualization Controls")
        vis_layout = QHBoxLayout(vis_grp)
        vis_layout.addWidget(QLabel("Deflection Scale Factor:"))
        self.sp_scale = QSpinBox()
        self.sp_scale.setRange(1, 500)
        self.sp_scale.setValue(50)
        self.sp_scale.valueChanged.connect(self.update_plot_only)
        vis_layout.addWidget(self.sp_scale)
        vis_layout.addStretch()
        layout.addWidget(vis_grp)
        
        tabs = QTabWidget()
        
        self.table_nodes = QTableWidget()
        self.table_nodes.setColumnCount(7)
        self.table_nodes.setHorizontalHeaderLabels(["Node ID", "Disp X (mm)", "Disp Y (mm)", "Disp Z (mm)", "Rx (rad)", "Ry (rad)", "Rz (rad)"])
        self.table_nodes.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tabs.addTab(self.table_nodes, "Nodal Displacements")
        
        self.table_elements = QTableWidget()
        self.table_elements.setColumnCount(7)
        self.table_elements.setHorizontalHeaderLabels(["Element ID", "Connectivity", "Axial Fx (N)", "Torsion Mx (N-mm)", "Bending My (N-mm)", "Bending Mz (N-mm)", "Max VM (MPa)"])
        self.table_elements.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tabs.addTab(self.table_elements, "Element Stresses")
        
        calc_tab = QWidget()
        calc_layout = QHBoxLayout(calc_tab)
        calc_layout.setContentsMargins(20, 20, 20, 20)
        
        self.calc_inputs_grp = QGroupBox("Interactive Parameters (mm, N-mm)")
        inputs_layout = QGridLayout(self.calc_inputs_grp)
        
        self.lbl_calc_m_name = QLabel("Bending Moment Mz (N-mm):")
        inputs_layout.addWidget(self.lbl_calc_m_name, 0, 0)
        self.calc_m = QSpinBox()
        self.calc_m.setRange(100000, 5000000)
        self.calc_m.setValue(1000000)
        self.calc_m.setSingleStep(50000)
        inputs_layout.addWidget(self.calc_m, 0, 1)
        
        self.slider_calc_m = QSlider(Qt.Horizontal)
        self.slider_calc_m.setRange(100000, 5000000)
        self.slider_calc_m.setValue(1000000)
        self.slider_calc_m.setSingleStep(50000)
        inputs_layout.addWidget(self.slider_calc_m, 1, 0, 1, 2)
        
        self.lbl_calc_rout_name = QLabel("Outer Radius (mm):")
        inputs_layout.addWidget(self.lbl_calc_rout_name, 2, 0)
        self.calc_rout = QDoubleSpinBox()
        self.calc_rout.setRange(10.0, 300.0)
        self.calc_rout.setValue(100.0)
        self.calc_rout.setSingleStep(1.0)
        inputs_layout.addWidget(self.calc_rout, 2, 1)
        
        self.slider_calc_rout = QSlider(Qt.Horizontal)
        self.slider_calc_rout.setRange(10, 300)
        self.slider_calc_rout.setValue(100)
        inputs_layout.addWidget(self.slider_calc_rout, 3, 0, 1, 2)
        
        self.lbl_calc_rin_name = QLabel("Inner Radius (mm):")
        inputs_layout.addWidget(self.lbl_calc_rin_name, 4, 0)
        self.calc_rin = QDoubleSpinBox()
        self.calc_rin.setRange(5.0, 290.0)
        self.calc_rin.setValue(80.0)
        self.calc_rin.setSingleStep(1.0)
        inputs_layout.addWidget(self.calc_rin, 4, 1)
        
        self.slider_calc_rin = QSlider(Qt.Horizontal)
        self.slider_calc_rin.setRange(5, 290)
        self.slider_calc_rin.setValue(80)
        inputs_layout.addWidget(self.slider_calc_rin, 5, 0, 1, 2)
        
        calc_layout.addWidget(self.calc_inputs_grp, 4)
        
        outputs_grp = QGroupBox("Analytical Results")
        outputs_layout = QVBoxLayout(outputs_grp)
        
        card_iz = QWidget()
        card_iz.setStyleSheet("background-color: #0b0f19; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05);")
        l_iz = QVBoxLayout(card_iz)
        t_iz = QLabel("Moment of Inertia (Iz)")
        t_iz.setObjectName("MetricTitle")
        t_iz.setAlignment(Qt.AlignCenter)
        self.lbl_calc_iz = QLabel("0.0")
        self.lbl_calc_iz.setObjectName("MetricValue")
        self.lbl_calc_iz.setAlignment(Qt.AlignCenter)
        self.lbl_calc_iz_unit = QLabel("mm⁴")
        self.lbl_calc_iz_unit.setStyleSheet("color: #9ca3af; font-size: 11px;")
        self.lbl_calc_iz_unit.setAlignment(Qt.AlignCenter)
        l_iz.addWidget(t_iz)
        l_iz.addWidget(self.lbl_calc_iz)
        l_iz.addWidget(self.lbl_calc_iz_unit)
        outputs_layout.addWidget(card_iz)
        
        card_sig = QWidget()
        card_sig.setStyleSheet("background-color: #0b0f19; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05);")
        l_sig = QVBoxLayout(card_sig)
        t_sig = QLabel("Bending Normal Stress")
        t_sig.setObjectName("MetricTitle")
        t_sig.setAlignment(Qt.AlignCenter)
        self.lbl_calc_sigma = QLabel("0.0")
        self.lbl_calc_sigma.setObjectName("MetricValue")
        self.lbl_calc_sigma.setAlignment(Qt.AlignCenter)
        self.lbl_calc_sigma_unit = QLabel("MPa")
        self.lbl_calc_sigma_unit.setStyleSheet("color: #9ca3af; font-size: 11px;")
        self.lbl_calc_sigma_unit.setAlignment(Qt.AlignCenter)
        l_sig.addWidget(t_sig)
        l_sig.addWidget(self.lbl_calc_sigma)
        l_sig.addWidget(self.lbl_calc_sigma_unit)
        outputs_layout.addWidget(card_sig)
        
        card_vm = QWidget()
        card_vm.setStyleSheet("background-color: #0b0f19; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05);")
        l_vm = QVBoxLayout(card_vm)
        t_vm = QLabel("Von Mises Stress")
        t_vm.setObjectName("MetricTitle")
        t_vm.setAlignment(Qt.AlignCenter)
        self.lbl_calc_vm = QLabel("0.0")
        self.lbl_calc_vm.setObjectName("MetricValue")
        self.lbl_calc_vm.setAlignment(Qt.AlignCenter)
        self.lbl_calc_vm_unit = QLabel("MPa")
        self.lbl_calc_vm_unit.setStyleSheet("color: #9ca3af; font-size: 11px;")
        self.lbl_calc_vm_unit.setAlignment(Qt.AlignCenter)
        l_vm.addWidget(t_vm)
        l_vm.addWidget(self.lbl_calc_vm)
        l_vm.addWidget(self.lbl_calc_vm_unit)
        outputs_layout.addWidget(card_vm)
        
        calc_layout.addWidget(outputs_grp, 3)
        tabs.addTab(calc_tab, "Stress Calculator")
        
        self.calc_m.valueChanged.connect(self.slider_calc_m.setValue)
        self.slider_calc_m.valueChanged.connect(self.calc_m.setValue)
        
        self.calc_rout.valueChanged.connect(self.on_calc_rout_spin_changed)
        self.slider_calc_rout.valueChanged.connect(self.on_calc_rout_slider_changed)
        
        self.calc_rin.valueChanged.connect(self.on_calc_rin_spin_changed)
        self.slider_calc_rin.valueChanged.connect(self.on_calc_rin_slider_changed)
        
        self.calc_m.valueChanged.connect(self.recalculate_analytical_stress)
        
        self.recalculate_analytical_stress()
        
        layout.addWidget(tabs)
        self.stacked_widget.addWidget(page)

    def setup_connections(self):
        self.combo_mat.currentIndexChanged.connect(self.on_material_change)
        self.combo_units.currentIndexChanged.connect(self.on_unit_system_changed)
        self.table_supports.itemChanged.connect(self.update_visualization)
        self.table_loads.itemChanged.connect(self.update_visualization)
        self.sp_elements.valueChanged.connect(self.update_visualization)
        
    def on_material_change(self):
        mat = self.combo_mat.currentText()
        is_metric = (self.unit_system == "Metric")
        
        if mat == "Structural Steel":
            E_val = 200000.0 if is_metric else 2.9e7
            yield_val = 250.0 if is_metric else 36000.0
            density_val = 7.85e-9 if is_metric else 0.284
            nu_val = 0.27
            
            self.sp_E.setValue(E_val); self.sp_nu.setValue(nu_val); self.sp_yield.setValue(yield_val); self.sp_density.setValue(density_val)
            self.sp_E.setEnabled(False); self.sp_nu.setEnabled(False); self.sp_yield.setEnabled(False); self.sp_density.setEnabled(False)
            
        elif mat == "Aluminum 6061-T6":
            E_val = 68900.0 if is_metric else 1.0e7
            yield_val = 276.0 if is_metric else 40000.0
            density_val = 2.70e-9 if is_metric else 0.098
            nu_val = 0.33
            
            self.sp_E.setValue(E_val); self.sp_nu.setValue(nu_val); self.sp_yield.setValue(yield_val); self.sp_density.setValue(density_val)
            self.sp_E.setEnabled(False); self.sp_nu.setEnabled(False); self.sp_yield.setEnabled(False); self.sp_density.setEnabled(False)
            
        elif mat == "Titanium Grade 5":
            E_val = 114000.0 if is_metric else 1.65e7
            yield_val = 880.0 if is_metric else 128000.0
            density_val = 4.43e-9 if is_metric else 0.160
            nu_val = 0.34
            
            self.sp_E.setValue(E_val); self.sp_nu.setValue(nu_val); self.sp_yield.setValue(yield_val); self.sp_density.setValue(density_val)
            self.sp_E.setEnabled(False); self.sp_nu.setEnabled(False); self.sp_yield.setEnabled(False); self.sp_density.setEnabled(False)
            
        else:
            self.sp_E.setEnabled(True); self.sp_nu.setEnabled(True); self.sp_yield.setEnabled(True); self.sp_density.setEnabled(True)

    # --- Interactive Stress Calculator Helpers ---
    def on_calc_rout_spin_changed(self, val):
        self.slider_calc_rout.blockSignals(True)
        self.slider_calc_rout.setValue(int(val))
        self.slider_calc_rout.blockSignals(False)
        self.recalculate_analytical_stress()
        
    def on_calc_rout_slider_changed(self, val):
        self.calc_rout.blockSignals(True)
        self.calc_rout.setValue(float(val))
        self.calc_rout.blockSignals(False)
        self.recalculate_analytical_stress()
        
    def on_calc_rin_spin_changed(self, val):
        self.slider_calc_rin.blockSignals(True)
        self.slider_calc_rin.setValue(int(val))
        self.slider_calc_rin.blockSignals(False)
        self.recalculate_analytical_stress()
        
    def on_calc_rin_slider_changed(self, val):
        self.calc_rin.blockSignals(True)
        self.calc_rin.setValue(float(val))
        self.calc_rin.blockSignals(False)
        self.recalculate_analytical_stress()

    def recalculate_analytical_stress(self):
        M = self.calc_m.value()
        ro = self.calc_rout.value()
        ri = self.calc_rin.value()
        
        # Enforce that inner radius is strictly smaller than outer radius
        if ri >= ro:
            ri = ro - 1.0
            self.calc_rin.blockSignals(True)
            self.calc_rin.setValue(ri)
            self.calc_rin.blockSignals(False)
            
            self.slider_calc_rin.blockSignals(True)
            self.slider_calc_rin.setValue(int(ri))
            self.slider_calc_rin.blockSignals(False)
            
        Iz = (np.pi / 4.0) * (ro**4 - ri**4)
        sigma = (M * ro) / Iz if Iz > 1e-12 else 0.0
        
        self.lbl_calc_iz.setText(f"{Iz:.2e}")
        self.lbl_calc_sigma.setText(f"{sigma:.2f}")
        self.lbl_calc_vm.setText(f"{sigma:.2f}")

    def scale_spinbox(self, spinbox, factor, min_val, max_val, decimals=None):
        spinbox.blockSignals(True)
        current_val = spinbox.value()
        spinbox.setRange(min(current_val, min_val), max(current_val, max_val))
        if decimals is not None:
            spinbox.setDecimals(decimals)
        
        new_val = current_val * factor
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(new_val)
        spinbox.blockSignals(False)

    def on_unit_system_changed(self, index):
        old_system = self.unit_system
        new_system = "Metric" if index == 0 else "Imperial"
        if old_system == new_system:
            return
            
        self.unit_system = new_system
        is_metric = (new_system == "Metric")
        
        len_factor = IN_TO_MM if is_metric else MM_TO_IN
        l_min, l_max = (100.0, 50000.0) if is_metric else (4.0, 2000.0)
        self.scale_spinbox(self.sp_length, len_factor, l_min, l_max)
        
        s_min, s_max = (0.1, 2000.0) if is_metric else (0.004, 80.0)
        shape_spins = [
            self.rt_do, self.rt_t, self.st_w, self.st_t, self.ret_w, self.ret_h, self.ret_t,
            self.ib_h, self.ib_w, self.ib_tf, self.ib_tw, self.rb_d, self.rebar_w, self.rebar_h,
            self.gen_ymax, self.gen_zmax
        ]
        for spin in shape_spins:
            self.scale_spinbox(spin, len_factor, s_min, s_max)
            
        a_factor = IN_TO_MM**2 if is_metric else MM_TO_IN**2
        a_min, a_max = (1.0, 1e8) if is_metric else (0.0015, 1.55e5)
        self.scale_spinbox(self.gen_a, a_factor, a_min, a_max)
        
        i_factor = IN_TO_MM**4 if is_metric else MM_TO_IN**4
        i_min, i_max = (1.0, 1e12) if is_metric else (2.4e-6, 2.4e6)
        for spin in [self.gen_iy, self.gen_iz, self.gen_j]:
            self.scale_spinbox(spin, i_factor, i_min, i_max)
            
        stress_factor = PSI_TO_MPA if is_metric else MPA_TO_PSI
        e_min, e_max = (100.0, 1000000.0) if is_metric else (14500.0, 1.45e8)
        self.scale_spinbox(self.sp_E, stress_factor, e_min, e_max)
        
        y_min, y_max_val = (1.0, 10000.0) if is_metric else (145.0, 1.45e6)
        self.scale_spinbox(self.sp_yield, stress_factor, y_min, y_max_val)
        
        dens_factor = LBM_IN3_TO_TONNE_MM3 if is_metric else TONNE_MM3_TO_LBM_IN3
        d_min, d_max = (1e-12, 1.0) if is_metric else (1e-5, 10.0)
        self.scale_spinbox(self.sp_density, dens_factor, d_min, d_max, decimals=12 if is_metric else 6)
        
        force_factor = LBF_TO_N if is_metric else N_TO_LBF
        f_min, f_max = (-1e9, 1e9) if is_metric else (-2.24e8, 2.24e8)
        for spin in [self.sp_fx, self.sp_fy, self.sp_fz]:
            self.scale_spinbox(spin, force_factor, f_min, f_max)
            
        mom_factor = LBFIN_TO_NMM if is_metric else NMM_TO_LBFIN
        m_min, m_max = (-1e9, 1e9) if is_metric else (-8.85e6, 8.85e6)
        self.scale_spinbox(self.sp_mx, mom_factor, m_min, m_max)
        
        self.calc_m.blockSignals(True)
        self.slider_calc_m.blockSignals(True)
        m_c_min, m_c_max = (100000, 5000000) if is_metric else (885, 44253)
        self.calc_m.setRange(m_c_min, m_c_max)
        self.slider_calc_m.setRange(m_c_min, m_c_max)
        new_calc_m_val = int(round(self.calc_m.value() * mom_factor))
        self.calc_m.setValue(new_calc_m_val)
        self.slider_calc_m.setValue(new_calc_m_val)
        self.calc_m.blockSignals(False)
        self.slider_calc_m.blockSignals(False)
        
        rout_min, rout_max = (10.0, 300.0) if is_metric else (0.4, 12.0)
        rin_min, rin_max = (5.0, 290.0) if is_metric else (0.2, 11.5)
        
        self.scale_spinbox(self.calc_rout, len_factor, rout_min, rout_max)
        self.scale_spinbox(self.calc_rin, len_factor, rin_min, rin_max)
        
        self.slider_calc_rout.blockSignals(True)
        self.slider_calc_rin.blockSignals(True)
        self.slider_calc_rout.setRange(int(rout_min), int(rout_max))
        self.slider_calc_rin.setRange(int(rin_min), int(rin_max))
        self.slider_calc_rout.setValue(int(round(self.calc_rout.value())))
        self.slider_calc_rin.setValue(int(round(self.calc_rin.value())))
        self.slider_calc_rout.blockSignals(False)
        self.slider_calc_rin.blockSignals(False)
        
        # Convert geometry node coordinates in self.model
        for node in self.model.geometry_nodes.values():
            node.point.x *= len_factor
            node.point.y *= len_factor
            node.point.z *= len_factor
            
        # Convert existing rows in table_loads
        self.table_loads.blockSignals(True)
        for r in range(self.table_loads.rowCount()):
            for c in range(1, 7):
                item = self.table_loads.item(r, c)
                if item is not None:
                    try:
                        val = float(item.text())
                        if c in [1, 2, 3]: # Force
                            val *= force_factor
                        else: # Moment
                            val *= mom_factor
                        item.setText(f"{val:.2f}")
                    except ValueError:
                        item.setText("0.0")
        self.table_loads.blockSignals(False)

        self.update_ui_labels()
        self.update_section_calculation()
        self.update_visualization()
        self.recalculate_analytical_stress()
        
        if self.U is not None:
            for i in range(len(self.nodes_3d)):
                self.U[i*6 + 0] *= len_factor
                self.U[i*6 + 1] *= len_factor
                self.U[i*6 + 2] *= len_factor
                
            self.nodes_3d *= len_factor
            self.max_model_stress *= stress_factor
            self.total_deflection *= len_factor
            
            for idx, res in self.element_results.items():
                res['von_mises']['max'] *= stress_factor
                f = res['local_forces']
                f[0] *= force_factor; f[1] *= force_factor; f[2] *= force_factor
                f[6] *= force_factor; f[7] *= force_factor; f[8] *= force_factor
                f[3] *= mom_factor; f[4] *= mom_factor; f[5] *= mom_factor
                f[9] *= mom_factor; f[10] *= mom_factor; f[11] *= mom_factor
                
            self.update_results_ui()

    def update_ui_labels(self):
        is_metric = (self.unit_system == "Metric")
        u_len = "mm" if is_metric else "in"
        u_stress = "MPa" if is_metric else "psi"
        u_force = "N" if is_metric else "lbf"
        u_mom = "N-mm" if is_metric else "lbf-in"
        u_dens = "tonne/mm³" if is_metric else "lbm/in³"
        
        self.lbl_sp_length.setText(f"Total Beam Length ({u_len}):")
        
        self.lbl_rt_do.setText(f"Outer Diameter ({u_len}):")
        self.lbl_rt_t.setText(f"Wall Thickness ({u_len}):")
        
        self.lbl_st_w.setText(f"Outer Width ({u_len}):")
        self.lbl_st_t.setText(f"Wall Thickness ({u_len}):")
        
        self.lbl_ret_w.setText(f"Outer Width ({u_len}):")
        self.lbl_ret_h.setText(f"Outer Height ({u_len}):")
        self.lbl_ret_t.setText(f"Wall Thickness ({u_len}):")
        
        self.lbl_ib_h.setText(f"Total Height ({u_len}):")
        self.lbl_ib_w.setText(f"Flange Width ({u_len}):")
        self.lbl_ib_tf.setText(f"Flange Thickness ({u_len}):")
        self.lbl_ib_tw.setText(f"Web Thickness ({u_len}):")
        
        self.lbl_rb_d.setText(f"Diameter ({u_len}):")
        
        self.lbl_rebar_w.setText(f"Width ({u_len}):")
        self.lbl_rebar_h.setText(f"Height ({u_len}):")
        
        self.lbl_gen_a.setText(f"Area (A) [{u_len}²]:")
        self.lbl_gen_iy.setText(f"Iy [{u_len}⁴]:")
        self.lbl_gen_iz.setText(f"Iz [{u_len}⁴]:")
        self.lbl_gen_j.setText(f"J [{u_len}⁴]:")
        self.lbl_gen_ymax.setText(f"y_max (fiber y) [{u_len}]:")
        self.lbl_gen_zmax.setText(f"z_max (fiber z) [{u_len}]:")
        
        self.lbl_E_name.setText(f"Young's Modulus ({u_stress}):")
        self.lbl_yield_name.setText(f"Yield Strength ({u_stress}):")
        self.lbl_density_name.setText(f"Density ({u_dens}):")
        
        self.lbl_fx_name.setText(f"Axial Fx ({u_force}):")
        self.lbl_fy_name.setText(f"Shear Fy ({u_force}):")
        self.lbl_fz_name.setText(f"Shear Fz ({u_force}):")
        self.lbl_mx_name.setText(f"Torsion Mx ({u_mom}):")
        
        self.lbl_metric_deflection_title.setText(f"Total Tip Deflection ({u_len})")
        self.lbl_metric_stress_title.setText(f"Max Von Mises ({u_stress})")
        
        self.lbl_calc_m_name.setText(f"Bending Moment Mz ({u_mom}):")
        self.lbl_calc_rout_name.setText(f"Outer Radius ({u_len}):")
        self.lbl_calc_rin_name.setText(f"Inner Radius ({u_len}):")
        
        self.lbl_calc_iz_unit.setText(f"{u_len}⁴")
        self.lbl_calc_sigma_unit.setText(f"{u_stress}")
        self.lbl_calc_vm_unit.setText(f"{u_stress}")
        
        self.calc_inputs_grp.setTitle(f"Interactive Parameters ({u_len}, {u_mom})")
        
        self.table_nodes.setHorizontalHeaderLabels([
            "Node ID", f"Disp X ({u_len})", f"Disp Y ({u_len})", f"Disp Z ({u_len})", 
            "Rx (rad)", "Ry (rad)", "Rz (rad)"
        ])
        self.table_elements.setHorizontalHeaderLabels([
            "Element ID", "Connectivity", f"Axial Fx ({u_force})", f"Torsion Mx ({u_mom})", 
            f"Bending My ({u_mom})", f"Bending Mz ({u_mom})", f"Max VM ({u_stress})"
        ])

    def log(self, msg):
        self.log_output.append(msg)
        QApplication.processEvents()

    # --- FEA Solver Logic ---
    def run_simulation(self):
        self.log_output.clear()
        self.log("[System] Initializing FEA solver...")
        
        self.recalculate_section_properties()
        
        self.get_preview_mesh()
        
        supports, loads = self.get_current_boundary_conditions()
        self.model.supports = supports
        self.model.loads = loads
        
        self.log(f"[BCs] Applied {len(supports)} supports and {len(loads)} point loads.")
        
        errors = self.model.validate_for_solving()
        if errors:
            self.log("[Validation Error] Cannot run solver:")
            for err in errors:
                self.log(f"  • {err}")
            return
            
        self.log("[Solve] Solving linear static system using WNFEA solver...")
        
        try:
            self.log("[Solver] Assembling global stiffness matrix and solving K * U = F...")
            U = solve_linear_static(self.model)
            
            self.log("[Solver] Computing element stresses and internal forces...")
            element_results = compute_element_stresses(self.model)
            
            self.log("[Solver] Analyzing result set...")
            rs = ResultSet(self.model)
            max_node, max_defl = rs.max_deflection()
            max_elem, max_vm = rs.max_von_mises()
            sf = rs.safety_factor()
            
            self.U = U
            self.nodes_3d = self.model.mesh_nodes
            self.elements_3d = self.model.mesh_elements
            self.element_results = self.model.element_results
            self.max_model_stress = max_vm
            self.total_deflection = max_defl
            self.safety_factor = sf
            
            self.log("[Success] Simulation completed successfully.")
            self.log(f"  • Max Deflection: {max_defl:.4e} {self.unit_system}")
            self.log(f"  • Max Von Mises: {max_vm:.3e} {self.unit_system}")
            self.log(f"  • Min Safety Factor: {sf:.2f}")
            
            self.update_results_ui()
            self.switch_page(4)
            
        except Exception as e:
            self.log(f"[Error] Solver failed: {e}")

    def update_results_ui(self):
        self.lbl_deflection.setText(f"{self.total_deflection:.4f}")
        self.lbl_stress.setText(f"{self.max_model_stress:.3f}")
        self.lbl_safety.setText(f"{self.safety_factor:.2f}")
        
        if self.safety_factor >= 1.5:
            self.lbl_safety.setStyleSheet("color: #22c55e;")
        elif self.safety_factor >= 1.0:
            self.lbl_safety.setStyleSheet("color: #eab308;")
        else:
            self.lbl_safety.setStyleSheet("color: #ef4444;")
            
        self.update_visualization()
        
        self.table_nodes.setRowCount(len(self.nodes_3d))
        for i in range(len(self.nodes_3d)):
            d = self.U[i*6:i*6+6]
            row_data = [str(i)] + [f"{v:.4f}" for v in d]
            for col, text in enumerate(row_data):
                self.table_nodes.setItem(i, col, QTableWidgetItem(text))
                
        self.table_elements.setRowCount(len(self.elements_3d))
        for i, res in self.element_results.items():
            f = res['local_forces']
            row_data = [
                str(i), f"{self.elements_3d[i][0]}->{self.elements_3d[i][1]}",
                f"{f[6]:.2f}", f"{f[9]:.2f}", f"{f[10]:.2f}", f"{f[11]:.2f}",
                f"{res['von_mises']['max']:.3f}"
            ]
            for col, text in enumerate(row_data):
                self.table_elements.setItem(i, col, QTableWidgetItem(text))

    def update_plot_only(self):
        self.update_visualization()

if __name__ == "__main__":
    # Workaround for high DPI displays
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    
    # Initialize material logic on boot
    window = FEAEngineApp()
    window.on_material_change()
    window.show()
    sys.exit(app.exec())
