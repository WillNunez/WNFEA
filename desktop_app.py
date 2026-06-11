import sys
import numpy as np
import pandas as pd
from scipy.linalg import solve
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

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
import re

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
        
        # --- Main Content (Stacked Widget) ---
        self.stacked_widget = QStackedWidget()
        self.splitter.addWidget(self.stacked_widget)
        
        # --- Right Documentation Panel ---
        self.doc_view = QWebEngineView()
        doc_settings = self.doc_view.settings()
        doc_settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        doc_settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        self.doc_view.setUrl(QUrl("file:///h:/Other%20computers/My%20Computer/Coding_Scratch/index.html"))
        self.splitter.addWidget(self.doc_view)
        
        # Set layout proportions
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 6)
        self.splitter.setStretchFactor(2, 4)
        
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setCollapsible(2, True)
        
        self.create_geometry_page()
        self.create_meshing_page()
        self.create_setup_page()
        self.create_solving_page()
        self.create_results_page()
        
        # Draw initial geometry preview
        self.update_geometry_preview()
        
        # Select first page
        self.switch_page(0)
        
    def switch_page(self, index):
        self.stacked_widget.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

    def toggle_doc_panel(self):
        visible = self.btn_toggle_doc.isChecked()
        self.doc_view.setVisible(visible)

    # --- Pages ---
    def create_geometry_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Left Panel - Controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        header = QLabel("📐 Geometry Setup")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        left_layout.addWidget(header)
        
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
        
        left_layout.addWidget(self.sec_grp)
        
        # 2. Option A: Parameterized Straight Beam
        self.param_grp = QGroupBox("Option A: Parameterized Straight Beam")
        g_layout = QGridLayout(self.param_grp)
        
        self.lbl_sp_length = QLabel("Total Beam Length (mm):")
        g_layout.addWidget(self.lbl_sp_length, 0, 0)
        self.sp_length = QDoubleSpinBox()
        self.sp_length.setRange(100.0, 50000.0)
        self.sp_length.setSingleStep(100.0)
        self.sp_length.setValue(2000.0)
        self.sp_length.valueChanged.connect(self.update_geometry_preview)
        g_layout.addWidget(self.sp_length, 0, 1)
        left_layout.addWidget(self.param_grp)
        
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
        
        left_layout.addWidget(self.step_grp)
        left_layout.addStretch()
        
        layout.addWidget(left_widget, 4)
        
        # Right Panel - 3D Geometry Preview
        preview_grp = QGroupBox("3D Geometry Preview")
        preview_layout = QVBoxLayout(preview_grp)
        self.geom_canvas = MatplotlibCanvas(self)
        preview_layout.addWidget(self.geom_canvas)
        
        layout.addWidget(preview_grp, 6)
        
        self.stacked_widget.addWidget(page)
        
        # Setup shape connections
        self.combo_shape.currentIndexChanged.connect(self.shape_inputs_stacked.setCurrentIndex)
        self.combo_shape.currentIndexChanged.connect(self.update_section_calculation)
        self.combo_shape.currentIndexChanged.connect(self.update_geometry_preview)
        
        # Link all sub-widgets valueChanged signals to update calculations
        for spin in [
            self.rt_do, self.rt_t, self.st_w, self.st_t, self.ret_w, self.ret_h, self.ret_t,
            self.ib_h, self.ib_w, self.ib_tf, self.ib_tw, self.rb_d, self.rebar_w, self.rebar_h,
            self.gen_a, self.gen_iy, self.gen_iz, self.gen_j, self.gen_ymax, self.gen_zmax
        ]:
            spin.valueChanged.connect(self.update_section_calculation)
            spin.valueChanged.connect(self.update_geometry_preview)

    def import_step_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open STEP File", "", "STEP Files (*.stp *.step)"
        )
        if not file_path:
            return
            
        edges = parse_step_file(file_path)
        if not edges:
            self.lbl_step_status.setText("❌ Failed to parse any curves/lines from STEP file.")
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
            # Target is mm
            if cad_unit == "m":
                scale = 1000.0
            elif cad_unit == "cm":
                scale = 10.0
            elif cad_unit == "in":
                scale = 25.4
        else:
            # Target is in
            if cad_unit == "m":
                scale = 1000.0 / 25.4
            elif cad_unit == "cm":
                scale = 10.0 / 25.4
            elif cad_unit == "mm":
                scale = 1.0 / 25.4
                
        nodes, elements = consolidate_edges(edges)
        nodes *= scale # Auto-scale to active system coordinates!
        
        self.step_edges = edges
        self.imported_nodes = nodes
        self.imported_elements = elements
        
        self.lbl_step_status.setText(
            f"✅ Loaded: {file_path.split('/')[-1]}\n"
            f"• CAD Unit: {cad_unit.upper()} (Auto-Scaled to {self.unit_system})\n"
            f"• Coords: {len(nodes)} unique vertices\n"
            f"• Edges: {len(elements)} parsed segments"
        )
        
        self.btn_clear_step.setVisible(True)
        self.param_grp.setEnabled(False)  # Disable parameterized inputs
        self.update_geometry_preview()
        
    def clear_step_import(self):
        self.step_edges = None
        self.imported_nodes = None
        self.imported_elements = None
        self.lbl_step_status.setText("No STEP file loaded. Using Parameterized Beam.")
        self.btn_clear_step.setVisible(False)
        self.param_grp.setEnabled(True)
        self.update_geometry_preview()

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

    def update_geometry_preview(self):
        self.geom_canvas.ax.clear()
        
        # Enforce fresh calculations to obtain latest ro and ri
        self.recalculate_section_properties()
        ro = max(self.y_max, self.z_max)
        shape = self.combo_shape.currentText()
        ri = ro * 0.8 if "Tube" in shape or "Beam" in shape else 0.0
        res = 12
        
        # Build segment pairs
        segments = []
        if self.step_edges is not None and len(self.imported_nodes) > 0:
            nodes = self.imported_nodes
            for idx, (n1, n2) in enumerate(self.imported_elements):
                segments.append((nodes[n1], nodes[n2]))
        else:
            L = self.sp_length.value()
            segments.append((np.array([0.0, 0.0, 0.0]), np.array([L, 0.0, 0.0])))
            
        # Draw 3D Volumetric Tubes/Cylinders
        for pt1, pt2 in segments:
            vec = pt2 - pt1
            mag = np.linalg.norm(vec)
            if mag < 1e-9: continue
            vn = vec / mag
            
            # Find orthogonal basis
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
            
            self.geom_canvas.ax.add_collection3d(
                Poly3DCollection(v_out, facecolor='#06b6d4', edgecolor='k', lw=0.15, alpha=0.8)
            )
            
            if ri > 0:
                cin = ri * (np.outer(np.cos(th), v1) + np.outer(np.sin(th), v2))
                v_in = []
                for j in range(res - 1):
                    v_in.append([pt1+cin[j], pt1+cin[j+1], pt2+cin[j+1], pt2+cin[j]])
                v_in.append([pt1+cin[res-1], pt1+cin[0], pt2+cin[0], pt2+cin[res-1]])
                self.geom_canvas.ax.add_collection3d(
                    Poly3DCollection(v_in, facecolor='gray', edgecolor='k', lw=0.1, alpha=0.3)
                )
                
        # Draw node markers at the joints
        all_pts = []
        if self.step_edges is not None and len(self.imported_nodes) > 0:
            all_pts = self.imported_nodes
        else:
            all_pts = np.array([[0.0, 0.0, 0.0], [self.sp_length.value(), 0.0, 0.0]])
        self.geom_canvas.ax.scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2], color='white', s=50)
        
        # Setup bounds
        if self.step_edges is not None and len(self.imported_nodes) > 0:
            nodes = self.imported_nodes
            min_bounds = nodes.min(axis=0)
            max_bounds = nodes.max(axis=0)
            centers = (min_bounds + max_bounds) / 2.0
            ranges = (max_bounds - min_bounds) / 2.0
            max_range = max(ranges.max(), 0.5)
            
            self.geom_canvas.ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
            self.geom_canvas.ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
            self.geom_canvas.ax.set_zlim(centers[2] - max_range, centers[2] + max_range)
        else:
            L = self.sp_length.value()
            self.geom_canvas.ax.set_xlim(L/2 - L*0.6, L/2 + L*0.6)
            self.geom_canvas.ax.set_ylim(-L*0.6, L*0.6)
            self.geom_canvas.ax.set_zlim(-L*0.6, L*0.6)
            
        u_len = "mm" if self.unit_system == "Metric" else "in"
        self.geom_canvas.ax.set_xlabel(f'X ({u_len})')
        self.geom_canvas.ax.set_ylabel(f'Y ({u_len})')
        self.geom_canvas.ax.set_zlabel(f'Z ({u_len})')
        self.geom_canvas.ax.set_box_aspect((1, 1, 1))
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
        layout.setContentsMargins(40, 40, 40, 40)
        
        header = QLabel("⚡ Problem Setup")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        layout.addWidget(header)
        
        # Materials (Steel in MPa)
        m_grp = QGroupBox("Material Properties")
        m_layout = QGridLayout(m_grp)
        
        self.combo_mat = QComboBox()
        self.combo_mat.addItems(["Structural Steel", "Aluminum 6061-T6", "Titanium Grade 5", "Custom"])
        m_layout.addWidget(QLabel("Preset:"), 0, 0)
        m_layout.addWidget(self.combo_mat, 0, 1)
        
        self.lbl_E_name = QLabel("Young's Modulus (MPa):")
        m_layout.addWidget(self.lbl_E_name, 1, 0)
        self.sp_E = QDoubleSpinBox()
        self.sp_E.setRange(100.0, 1000000.0)
        self.sp_E.setValue(200000.0)
        m_layout.addWidget(self.sp_E, 1, 1)
        
        m_layout.addWidget(QLabel("Poisson's Ratio:"), 2, 0)
        self.sp_nu = QDoubleSpinBox()
        self.sp_nu.setRange(0.0, 0.49)
        self.sp_nu.setSingleStep(0.01)
        self.sp_nu.setValue(0.27)
        m_layout.addWidget(self.sp_nu, 2, 1)
        
        self.lbl_yield_name = QLabel("Yield Strength (MPa):")
        m_layout.addWidget(self.lbl_yield_name, 3, 0)
        self.sp_yield = QDoubleSpinBox()
        self.sp_yield.setRange(1.0, 10000.0)
        self.sp_yield.setValue(250.0)
        m_layout.addWidget(self.sp_yield, 3, 1)
        
        self.lbl_density_name = QLabel("Density (tonne/mm³):")
        m_layout.addWidget(self.lbl_density_name, 4, 0)
        self.sp_density = QDoubleSpinBox()
        self.sp_density.setDecimals(12)
        self.sp_density.setRange(1e-12, 1.0)
        self.sp_density.setValue(7.85e-9)
        self.sp_density.setSingleStep(1e-9)
        m_layout.addWidget(self.sp_density, 4, 1)
        
        layout.addWidget(m_grp)
        
        # Loads in N and N-mm
        l_grp = QGroupBox("Tip Point Loads")
        l_layout = QGridLayout(l_grp)
        
        self.sp_fx = QDoubleSpinBox()
        self.sp_fx.setRange(-1e9, 1e9)
        self.sp_fx.setValue(0.0)
        self.lbl_fx_name = QLabel("Axial Fx (N):")
        l_layout.addWidget(self.lbl_fx_name, 0, 0)
        l_layout.addWidget(self.sp_fx, 0, 1)
        
        self.sp_fy = QDoubleSpinBox()
        self.sp_fy.setRange(-1e9, 1e9)
        self.sp_fy.setValue(-1000.0)
        self.lbl_fy_name = QLabel("Shear Fy (N):")
        l_layout.addWidget(self.lbl_fy_name, 1, 0)
        l_layout.addWidget(self.sp_fy, 1, 1)
        
        self.sp_fz = QDoubleSpinBox()
        self.sp_fz.setRange(-1e9, 1e9)
        self.sp_fz.setValue(0.0)
        self.lbl_fz_name = QLabel("Shear Fz (N):")
        l_layout.addWidget(self.lbl_fz_name, 2, 0)
        l_layout.addWidget(self.sp_fz, 2, 1)
        
        self.sp_mx = QDoubleSpinBox()
        self.sp_mx.setRange(-1e9, 1e9)
        self.sp_mx.setValue(0.0)
        self.lbl_mx_name = QLabel("Torsion Mx (N-mm):")
        l_layout.addWidget(self.lbl_mx_name, 3, 0)
        l_layout.addWidget(self.sp_mx, 3, 1)
        
        layout.addWidget(l_grp)
        layout.addStretch()
        self.stacked_widget.addWidget(page)

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
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Metrics Top Bar (Consistent Units)
        metrics_layout = QHBoxLayout()
        
        # Tip Deflection Card
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
        
        # Stress Card
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
        
        # Safety Factor Card
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
        
        # Visual & Data Splitting
        tabs = QTabWidget()
        
        # Plot Tab
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        
        vis_controls = QHBoxLayout()
        vis_controls.addWidget(QLabel("Deflection Scale Factor:"))
        self.sp_scale = QSpinBox()
        self.sp_scale.setRange(1, 500)
        self.sp_scale.setValue(50)
        self.sp_scale.valueChanged.connect(self.update_plot_only)
        vis_controls.addWidget(self.sp_scale)
        vis_controls.addStretch()
        plot_layout.addLayout(vis_controls)
        
        self.canvas = MatplotlibCanvas(self)
        plot_layout.addWidget(self.canvas)
        tabs.addTab(plot_tab, "3D Visualization")
        
        # Nodes Tab
        self.table_nodes = QTableWidget()
        self.table_nodes.setColumnCount(7)
        self.table_nodes.setHorizontalHeaderLabels(["Node ID", "Disp X (mm)", "Disp Y (mm)", "Disp Z (mm)", "Rx (rad)", "Ry (rad)", "Rz (rad)"])
        self.table_nodes.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tabs.addTab(self.table_nodes, "Nodal Displacements")
        
        # Elements Tab
        self.table_elements = QTableWidget()
        self.table_elements.setColumnCount(7)
        self.table_elements.setHorizontalHeaderLabels(["Element ID", "Connectivity", "Axial Fx (N)", "Torsion Mx (N-mm)", "Bending My (N-mm)", "Bending Mz (N-mm)", "Max VM (MPa)"])
        self.table_elements.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tabs.addTab(self.table_elements, "Element Stresses")
        
        # --- Interactive Stress Calculator Tab in mm and N-mm ---
        calc_tab = QWidget()
        calc_layout = QHBoxLayout(calc_tab)
        calc_layout.setContentsMargins(20, 20, 20, 20)
        
        # Left side inputs
        self.calc_inputs_grp = QGroupBox("Interactive Parameters (mm, N-mm)")
        inputs_layout = QGridLayout(self.calc_inputs_grp)
        
        # Bending Moment Slider & Spinbox
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
        
        # Outer Radius Slider & Spinbox
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
        
        # Inner Radius Slider & Spinbox
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
        
        # Right side outputs
        outputs_grp = QGroupBox("Analytical Results")
        outputs_layout = QVBoxLayout(outputs_grp)
        
        # Moment of inertia output card
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
        
        # Bending stress output card
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
        
        # Von Mises output card
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
        
        # Connect interactive stress calculator signals
        self.calc_m.valueChanged.connect(self.slider_calc_m.setValue)
        self.slider_calc_m.valueChanged.connect(self.calc_m.setValue)
        
        self.calc_rout.valueChanged.connect(self.on_calc_rout_spin_changed)
        self.slider_calc_rout.valueChanged.connect(self.on_calc_rout_slider_changed)
        
        self.calc_rin.valueChanged.connect(self.on_calc_rin_spin_changed)
        self.slider_calc_rin.valueChanged.connect(self.on_calc_rin_slider_changed)
        
        self.calc_m.valueChanged.connect(self.recalculate_analytical_stress)
        
        # Run initial calculation
        self.recalculate_analytical_stress()
        
        layout.addWidget(tabs)
        self.stacked_widget.addWidget(page)

    def setup_connections(self):
        self.combo_mat.currentIndexChanged.connect(self.on_material_change)
        self.combo_units.currentIndexChanged.connect(self.on_unit_system_changed)
        
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
        
        # --- Length conversions ---
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
            
        # General Area (mm² <-> in²)
        a_factor = IN_TO_MM**2 if is_metric else MM_TO_IN**2
        a_min, a_max = (1.0, 1e8) if is_metric else (0.0015, 1.55e5)
        self.scale_spinbox(self.gen_a, a_factor, a_min, a_max)
        
        # General Inertia (mm⁴ <-> in⁴)
        i_factor = IN_TO_MM**4 if is_metric else MM_TO_IN**4
        i_min, i_max = (1.0, 1e12) if is_metric else (2.4e-6, 2.4e6)
        for spin in [self.gen_iy, self.gen_iz, self.gen_j]:
            self.scale_spinbox(spin, i_factor, i_min, i_max)
            
        # --- Stress / Elastic Modulus conversions ---
        stress_factor = PSI_TO_MPA if is_metric else MPA_TO_PSI
        e_min, e_max = (100.0, 1000000.0) if is_metric else (14500.0, 1.45e8)
        self.scale_spinbox(self.sp_E, stress_factor, e_min, e_max)
        
        y_min, y_max_val = (1.0, 10000.0) if is_metric else (145.0, 1.45e6)
        self.scale_spinbox(self.sp_yield, stress_factor, y_min, y_max_val)
        
        # --- Density conversions ---
        dens_factor = LBM_IN3_TO_TONNE_MM3 if is_metric else TONNE_MM3_TO_LBM_IN3
        d_min, d_max = (1e-12, 1.0) if is_metric else (1e-5, 10.0)
        self.scale_spinbox(self.sp_density, dens_factor, d_min, d_max, decimals=12 if is_metric else 6)
        
        # --- Force conversions ---
        force_factor = LBF_TO_N if is_metric else N_TO_LBF
        f_min, f_max = (-1e9, 1e9) if is_metric else (-2.24e8, 2.24e8)
        for spin in [self.sp_fx, self.sp_fy, self.sp_fz]:
            self.scale_spinbox(spin, force_factor, f_min, f_max)
            
        # --- Moment conversions ---
        mom_factor = LBFIN_TO_NMM if is_metric else NMM_TO_LBFIN
        m_min, m_max = (-1e9, 1e9) if is_metric else (-8.85e6, 8.85e6)
        self.scale_spinbox(self.sp_mx, mom_factor, m_min, m_max)
        
        # --- Interactive Stress Calculator conversions ---
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
        
        # Convert parsed CAD node coordinates if they exist
        if self.imported_nodes is not None and len(self.imported_nodes) > 0:
            self.imported_nodes *= len_factor

        # Update Table Headers & Labels
        self.update_ui_labels()
        
        self.update_section_calculation()
        self.update_geometry_preview()
        self.recalculate_analytical_stress()
        
        # Converted active FEA Results if they exist
        if self.U is not None:
            # U conversions (translations by len_factor, rotations unchanged)
            for i in range(len(self.nodes_3d)):
                self.U[i*6 + 0] *= len_factor
                self.U[i*6 + 1] *= len_factor
                self.U[i*6 + 2] *= len_factor
                
            self.nodes_3d *= len_factor
            self.max_model_stress *= stress_factor
            self.total_deflection *= len_factor
            
            for idx, res in self.element_results.items():
                res['von_mises_stresses']['max'] *= stress_factor
                f = res['local_forces_moments']
                # forces
                f[0] *= force_factor; f[1] *= force_factor; f[2] *= force_factor
                f[6] *= force_factor; f[7] *= force_factor; f[8] *= force_factor
                # moments
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
        
        # Force a fresh section calculation to ensure variables are in sync
        self.recalculate_section_properties()
        
        beam_length = self.sp_length.value()
        num_elements = self.sp_elements.value()
        E = self.sp_E.value()
        nu = self.sp_nu.value()
        yield_strength = self.sp_yield.value()
        
        fx, fy, fz, mx = self.sp_fx.value(), self.sp_fy.value(), self.sp_fz.value(), self.sp_mx.value()
        
        cs = {
            "Area": self.A, 
            "Iy": self.Iy, 
            "Iz": self.Iz, 
            "J": self.J, 
            "y_max": self.y_max, 
            "z_max": self.z_max
        }
        mat = {"youngs_modulus": E, "poissons_ratio": nu}
        
        # Mesh Generation
        u_len = "mm" if self.unit_system == "Metric" else "in"
        self.log(f"[Mesh] Generating {num_elements} elements over {beam_length} {u_len}...")
        x_coords = np.linspace(0.0, beam_length, num_elements + 1)
        self.nodes_3d = np.zeros((num_elements + 1, 3))
        self.nodes_3d[:, 0] = x_coords
        self.elements_3d = np.array([[i, i+1] for i in range(num_elements)])
        
        num_nodes = len(self.nodes_3d)
        K = np.zeros((num_nodes*6, num_nodes*6))
        F = np.zeros(num_nodes*6)
        
        # Apply Loads
        tip = num_nodes - 1
        F[tip*6 + 0] = fx
        F[tip*6 + 1] = fy
        F[tip*6 + 2] = fz
        F[tip*6 + 3] = mx
        
        self.log("[Assemble] Building global stiffness matrix...")
        for n1, n2 in self.elements_3d:
            Ke = self.calculate_ke(self.nodes_3d[n1], self.nodes_3d[n2], mat, cs)
            dofs = np.concatenate([np.arange(n1*6, n1*6+6), np.arange(n2*6, n2*6+6)])
            for i in range(12):
                for j in range(12):
                    K[dofs[i], dofs[j]] += Ke[i, j]
                    
        # Apply BCs
        for dof in range(6):
            K[dof, :] = 0; K[:, dof] = 0; K[dof, dof] = 1; F[dof] = 0
            
        # Solve
        self.log("[Solve] Inverting stiffness matrix...")
        try:
            self.U = solve(K, F)
            self.log("[Success] Displacements computed.")
        except Exception as e:
            self.log(f"[Error] Solver failed: {e}")
            return
            
        # Post-Processing
        self.element_results = {}
        self.max_model_stress = 0.0
        
        self.log("[Post] Calculating elemental stresses...")
        for idx, (n1, n2) in enumerate(self.elements_3d):
            res = self.calculate_stress(self.nodes_3d[n1], self.nodes_3d[n2], [n1, n2], self.U, mat, cs)
            self.element_results[idx] = res
            if res['von_mises_stresses']['max'] > self.max_model_stress:
                self.max_model_stress = res['von_mises_stresses']['max']
                
        tip_dx, tip_dy, tip_dz = self.U[tip*6+0], self.U[tip*6+1], self.U[tip*6+2]
        self.total_deflection = np.sqrt(tip_dx**2 + tip_dy**2 + tip_dz**2)
        self.safety_factor = yield_strength / self.max_model_stress if self.max_model_stress > 1e-3 else 999.0
        
        self.log("[Finished] Simulation complete. Navigating to results...")
        self.update_results_ui()
        self.switch_page(4)  # Go to results tab

    def calculate_ke(self, n1, n2, mat, cs):
        E, G = mat['youngs_modulus'], mat['youngs_modulus'] / (2*(1+mat['poissons_ratio']))
        A, Iy, Iz, J = cs['Area'], cs['Iy'], cs['Iz'], cs['J']
        L = np.linalg.norm(n2 - n1)
        if L < 1e-9: return np.zeros((12, 12))
        
        K = np.zeros((12, 12))
        axial = E*A/L
        K[0,0] = K[6,6] = axial; K[0,6] = K[6,0] = -axial
        torsion = G*J/L
        K[3,3] = K[9,9] = torsion; K[3,9] = K[9,3] = -torsion
        bz1, bz2, bz3, bz4 = 12*E*Iz/L**3, 6*E*Iz/L**2, 4*E*Iz/L, 2*E*Iz/L
        K[1,1] = K[7,7] = bz1; K[1,7] = K[7,1] = -bz1
        K[1,5] = K[5,1] = K[1,11] = K[11,1] = bz2
        K[5,5] = K[11,11] = bz3; K[5,11] = K[11,5] = bz4
        K[5,7] = K[7,5] = K[7,11] = K[11,7] = -bz2
        by1, by2, by3, by4 = 12*E*Iy/L**3, 6*E*Iy/L**2, 4*E*Iy/L, 2*E*Iy/L
        K[2,2] = K[8,8] = by1; K[2,8] = K[8,2] = -by1
        K[2,4] = K[4,2] = K[2,10] = K[10,2] = -by2
        K[4,4] = K[10,10] = by3; K[4,10] = K[10,4] = by4
        K[4,8] = K[8,4] = K[8,10] = K[10,8] = by2
        return K

    def calculate_stress(self, p1, p2, indices, U, mat, cs):
        E, nu = mat['youngs_modulus'], mat['poissons_ratio']
        A, Iy, Iz, J = cs['Area'], cs['Iy'], cs['Iz'], cs['J']
        y_max, z_max = cs['y_max'], cs['z_max']
        L = np.linalg.norm(p2 - p1)
        if L < 1e-9: return {'local_forces_moments': np.zeros(12), 'von_mises_stresses': {'max': 0.0}}
        
        vec = (p2 - p1) / L
        R = np.eye(3)
        T_node = np.zeros((6,6)); T_node[:3,:3] = T_node[3:,3:] = R
        T = np.zeros((12,12)); T[:6,:6] = T[6:12,6:12] = T_node
        
        dofs = [idx*6+i for idx in indices for i in range(6)]
        u_local = T @ U[dofs]
        f_local = self.calculate_ke(p1, p2, mat, cs) @ u_local
        
        Fx2, Mx2, My2, Mz2 = f_local[6], f_local[9], f_local[10], f_local[11]
        
        vm_list = []
        # Check stresses at extreme points
        for yp, zp in [[y_max, 0.0], [-y_max, 0.0], [0.0, z_max], [0.0, -z_max]]:
            sig = Fx2/A + (-Mz2*yp)/Iz + (My2*zp)/Iy if Iz > 1e-12 and Iy > 1e-12 else Fx2/A
            r_outer = max(y_max, z_max)
            tau = (np.abs(Mx2)*r_outer)/J if J > 1e-12 else 0.0
            vm_list.append(np.sqrt(sig**2 + 3.0*tau**2))
            
        return {'local_forces_moments': f_local, 'von_mises_stresses': {'max': max(vm_list)}}

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
            
        self.update_plot_only()
        
        # Populate Tables (Direct N-mm and mm units)
        self.table_nodes.setRowCount(len(self.nodes_3d))
        for i in range(len(self.nodes_3d)):
            d = self.U[i*6:i*6+6]
            row_data = [str(i)] + [f"{v:.4f}" for v in d]
            for col, text in enumerate(row_data):
                self.table_nodes.setItem(i, col, QTableWidgetItem(text))
                
        self.table_elements.setRowCount(len(self.elements_3d))
        for i, res in self.element_results.items():
            f = res['local_forces_moments']
            row_data = [
                str(i), f"{self.elements_3d[i][0]}->{self.elements_3d[i][1]}",
                f"{f[6]:.2f}", f"{f[9]:.2f}", f"{f[10]:.2f}", f"{f[11]:.2f}",
                f"{res['von_mises_stresses']['max']:.3f}"
            ]
            for col, text in enumerate(row_data):
                self.table_elements.setItem(i, col, QTableWidgetItem(text))

    def update_plot_only(self):
        if self.U is None: return
        self.canvas.ax.clear()
        
        sf = self.sp_scale.value()
        defs = self.U.reshape(-1, 6)
        deformed = self.nodes_3d + defs[:, :3] * sf
        
        all_stresses = [res['von_mises_stresses']['max'] for res in self.element_results.values()]
        vmin, vmax = min(all_stresses), max(all_stresses)
        if vmax - vmin < 1e-3: vmin -= 1e3; vmax += 1e3
        norm = plt.Normalize(vmin, vmax)
        cmap = plt.cm.jet
        
        ro = max(self.y_max, self.z_max)
        shape = self.combo_shape.currentText()
        ri = ro * 0.8 if "Tube" in shape or "Beam" in shape else 0.0
        res = 12
        
        for i, (n1, n2) in enumerate(self.elements_3d):
            pt1, pt2 = deformed[n1], deformed[n2]
            c = cmap(norm(all_stresses[i]))
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
            
            self.canvas.ax.add_collection3d(Poly3DCollection(v_out, facecolor=c, edgecolor='k', lw=0.15, alpha=0.9))
            
            if ri > 0:
                cin = ri * (np.outer(np.cos(th), v1) + np.outer(np.sin(th), v2))
                v_in = []
                for j in range(res - 1):
                    v_in.append([pt1+cin[j], pt1+cin[j+1], pt2+cin[j+1], pt2+cin[j]])
                v_in.append([pt1+cin[res-1], pt1+cin[0], pt2+cin[0], pt2+cin[res-1]])
                self.canvas.ax.add_collection3d(Poly3DCollection(v_in, facecolor='gray', edgecolor='k', lw=0.1, alpha=0.4))
            
        self.canvas.ax.scatter(deformed[:,0], deformed[:,1], deformed[:,2], color='white', s=50)
        
        L = self.sp_length.value()
        self.canvas.ax.set_xlim(L/2 - L*0.6, L/2 + L*0.6)
        self.canvas.ax.set_ylim(-L*0.6, L*0.6)
        self.canvas.ax.set_zlim(-L*0.6, L*0.6)
        u_len = "mm" if self.unit_system == "Metric" else "in"
        self.canvas.ax.set_xlabel(f'X ({u_len})')
        self.canvas.ax.set_ylabel(f'Y ({u_len})')
        self.canvas.ax.set_zlabel(f'Z ({u_len})')
        self.canvas.ax.set_box_aspect((1, 1, 1))
        self.canvas.draw()

if __name__ == "__main__":
    # Workaround for high DPI displays
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    
    # Initialize material logic on boot
    window = FEAEngineApp()
    window.on_material_change()
    window.show()
    sys.exit(app.exec())
