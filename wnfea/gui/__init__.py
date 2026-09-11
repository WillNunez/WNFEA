"""
WNFEA Web GUI & Interactive Generative Design Studio.
----------------------------------------------------
Provides a local web interface for parameter input, material selection,
topology optimization targets, manufacturability constraints (3-axis CNC),
and real-time 3D WebGL visualization.
"""

from .server import GUIServer, start_gui_server

__all__ = ["GUIServer", "start_gui_server"]
