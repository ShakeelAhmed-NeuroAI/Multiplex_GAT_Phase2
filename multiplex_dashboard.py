import sys
import time
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QLabel
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg

class MultiplexRealTimeDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Phase 2: Multiplex Cross-Frequency Graph Attention Telemetry Console")
        self.resize(1200, 800)
        
        # Initialize simulation telemetry tracking data
        self.time_steps = list(range(50))
        self.latency_history = list(np.random.normal(0.1429, 0.01, 50))
        self.u_intensity_history = list(np.zeros(50))
        
        # Initialize primary core structural window widgets
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QGridLayout(central_widget)
        
        # 1. Add status configuration header strings
        self.status_label = QLabel("SYSTEM STATUS: ACTIVE OPEN-LOOP EMULATION | MEAN HARDWARE LATENCY: 0.1429 ms")
        self.status_label.setStyleSheet("font-weight: bold; color: #00FF00; background-color: #111; padding: 10px; font-family: Consolas;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label, 0, 0, 1, 3)
        
        # 2. Configure 3D Multiplex Matrix Image Viewers
        # Layer 0 View
        self.win_l0 = pg.ImageView()
        self.win_l0.ui.histogram.hide()
        self.win_l0.ui.roiBtn.hide()
        self.win_l0.ui.menuBtn.hide()
        main_layout.addWidget(self.create_card("Layer 0: Theta-Beta Connectome Matrix", self.win_l0), 1, 0)
        
        # Layer 1 View
        self.win_l1 = pg.ImageView()
        self.win_l1.ui.histogram.hide()
        self.win_l1.ui.roiBtn.hide()
        self.win_l1.ui.menuBtn.hide()
        main_layout.addWidget(self.create_card("Layer 1: Alpha-Gamma Connectome Matrix", self.win_l1), 1, 1)
        
        # Layer 2 View
        self.win_l2 = pg.ImageView()
        self.win_l2.ui.histogram.hide()
        self.win_l2.ui.roiBtn.hide()
        self.win_l2.ui.menuBtn.hide()
        main_layout.addWidget(self.create_card("Layer 2: Delta-Alpha Connectome Matrix", self.win_l2), 1, 2)
        
        # 3. Configure Real-Time Telemetry Line Charts
        # Hardware Execution Latency Tracking Chart
        self.latency_plot = pg.PlotWidget()
        self.latency_plot.setBackground('#1e1e1e')
        self.latency_plot.setLabel('left', 'Execution Latency', units='ms')
        self.latency_plot.setLabel('bottom', 'Inference Iterations')
        self.latency_curve = self.latency_plot.plot(self.latency_history, pen=pg.mkPen(color='#FF5555', width=2))
        main_layout.addWidget(self.create_card("On-Edge INT8 Processing Latency Monitor", self.latency_plot), 2, 0, 1, 1)
        
        # Control Feedback Intensity Tracking Chart
        self.control_plot = pg.PlotWidget()
        self.control_plot.setBackground('#1e1e1e')
        self.control_plot.setLabel('left', 'Modulation Feedback Intensity', units='uV')
        self.control_plot.setLabel('bottom', 'Execution Epochs')
        self.control_curve = self.control_plot.plot(self.u_intensity_history, pen=pg.mkPen(color='#55FF55', width=2))
        main_layout.addWidget(self.create_card("Safety-Constrained Slew Clamped Output Track", self.control_plot), 2, 1, 1, 2)
        
        # 4. Initialize real-time continuous refresh update timer loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_telemetry_frame)
        self.timer.start(100) # Refresh windows every 100 milliseconds
        
    def create_card(self, title_text, widget):
        card = QWidget()
        layout = QVBoxLayout(card)
        label = QLabel(title_text)
        label.setStyleSheet("font-size: 13px; font-weight: bold; color: #FFF; font-family: Arial;")
        layout.addWidget(label)
        layout.addWidget(widget)
        card.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333; border-radius: 4px; padding: 5px;")
        return card
        
    def update_telemetry_frame(self):
        # 1. Simulate real-time continuous fluctuations across multiplex matrices
        # Apply a threshold mock logic to emulate your noise gated tau baseline
        l0_mat = np.random.rand(32, 32)
        l0_mat[l0_mat < 0.45] = 0.0
        
        l1_mat = np.random.rand(32, 32)
        l1_mat[l1_mat < 0.45] = 0.0
        
        l2_mat = np.random.rand(32, 32)
        l2_mat[l2_mat < 0.45] = 0.0
        
        # Update image elements safely
        self.win_l0.setImage(l0_mat, autoLevels=True)
        self.win_l1.setImage(l1_mat, autoLevels=True)
        self.win_l2.setImage(l2_mat, autoLevels=True)
        
        # 2. Update execution hardware latency charts sequentially
        self.latency_history.pop(0)
        self.latency_history.append(np.random.normal(0.1429, 0.008))
        self.latency_curve.setData(self.latency_history)
        
        # 3. Update continuous closed-loop control modulation parameters
        self.u_intensity_history.pop(0)
        # Emulate tracking oscillations targeting your r(t) bounds
        self.u_intensity_history.append(float(np.sin(time.time()) * 0.35))
        self.control_curve.setData(self.u_intensity_history)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    console = MultiplexRealTimeDashboard()
    console.show()
    sys.exit(app.exec())
