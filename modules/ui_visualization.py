import sys
import math
import numpy as np
import pyaudio
import threading
import logging
from typing import Optional, Tuple
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt5.QtGui import QPainter, QColor, QPainterPath, QFont, QPen

logger = logging.getLogger(__name__)

class PulsingLight(QWidget):
    """Sound-responsive pulsing light visualization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Audio processing settings
        self.chunk_size = 1024
        self.sample_format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 44100
        
        # Visualization settings
        self.base_radius = 100
        self.max_radius = 200
        self.current_radius = self.base_radius
        self.color = QColor(29, 185, 84)  # Soft green color
        self.pulse_speed = 0.1
        self.audio_multiplier = 5.0
        
        # Animation settings
        self.angle = 0
        self.particles = []
        self.max_particles = 50
        
        # Initialize audio processing
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_thread = None
        self.is_listening = False
        
        # Setup update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.setInterval(16)  # ~60 FPS
        self.timer.start()
        
        # Set window properties
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Initialize audio visualization
        self.start_audio_processing()
    
    def start_audio_processing(self):
        """Start audio processing in a separate thread"""
        try:
            self.stream = self.audio.open(
                format=self.sample_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_listening = True
            self.audio_thread = threading.Thread(target=self._process_audio)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start audio processing: {e}")
    
    def stop_audio_processing(self):
        """Stop audio processing"""
        self.is_listening = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
    
    def _process_audio(self):
        """Process audio input and update visualization"""
        while self.is_listening:
            try:
                data = np.frombuffer(
                    self.stream.read(self.chunk_size, exception_on_overflow=False),
                    dtype=np.float32
                )
                
                # Calculate audio level
                level = np.abs(data).mean()
                
                # Update visualization parameters
                target_radius = self.base_radius + (level * self.audio_multiplier * 100)
                self.current_radius += (target_radius - self.current_radius) * self.pulse_speed
                self.current_radius = min(self.max_radius, max(self.base_radius, self.current_radius))
                
                # Add particles based on audio level
                if level > 0.01:
                    self._add_particle()
                
                # Increment angle for ripple effect
                self.angle += 0.05
                
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                break
    
    def _add_particle(self):
        """Add a new particle to the visualization"""
        if len(self.particles) < self.max_particles:
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(2, 5)
            size = np.random.uniform(2, 6)
            lifetime = np.random.uniform(0.5, 2.0)
            
            self.particles.append({
                'x': 0,
                'y': 0,
                'angle': angle,
                'speed': speed,
                'size': size,
                'lifetime': lifetime,
                'age': 0
            })
    
    def _update_particles(self):
        """Update particle positions and lifetimes"""
        for particle in self.particles[:]:
            particle['age'] += 0.016  # ~60 FPS
            
            if particle['age'] >= particle['lifetime']:
                self.particles.remove(particle)
                continue
                
            # Update position
            particle['x'] += math.cos(particle['angle']) * particle['speed']
            particle['y'] += math.sin(particle['angle']) * particle['speed']
    
    def paintEvent(self, event):
        """Paint the visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate center position
        center = QPoint(self.width() // 2, self.height() // 2)
        
        # Draw background glow
        gradient = self._create_radial_gradient(
            center,
            self.current_radius * 1.5
        )
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(
            center,
            int(self.current_radius * 1.5),
            int(self.current_radius * 1.5)
        )
        
        # Draw main circle
        painter.setBrush(self.color)
        painter.drawEllipse(
            center,
            int(self.current_radius),
            int(self.current_radius)
        )
        
        # Draw particles
        self._update_particles()
        for particle in self.particles:
            alpha = 255 * (1 - (particle['age'] / particle['lifetime']))
            color = QColor(self.color)
            color.setAlpha(int(alpha))
            
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            
            x = center.x() + particle['x']
            y = center.y() + particle['y']
            size = particle['size']
            
            painter.drawEllipse(
                QPoint(int(x), int(y)),
                int(size),
                int(size)
            )
        
        # Draw ripple effect
        self.angle += 0.05
        for i in range(3):
            phase = self.angle + (i * math.pi * 2 / 3)
            ripple_radius = self.current_radius + (math.sin(phase) * 20)
            
            pen = QPen(self.color)
            pen.setWidth(2)
            alpha = 127 + (math.sin(phase) * 128)
            pen.setColor(QColor(self.color.red(), self.color.green(),
                              self.color.blue(), int(alpha)))
            
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(
                center,
                int(ripple_radius),
                int(ripple_radius)
            )
    
    def _create_radial_gradient(self, center: QPoint, radius: float):
        """Create a radial gradient for the glow effect"""
        from PyQt5.QtGui import QRadialGradient
        gradient = QRadialGradient(center, radius)
        
        color = self.color
        gradient.setColorAt(0, QColor(color.red(), color.green(),
                                    color.blue(), 127))
        gradient.setColorAt(0.5, QColor(color.red(), color.green(),
                                      color.blue(), 64))
        gradient.setColorAt(1, QColor(color.red(), color.green(),
                                    color.blue(), 0))
        
        return gradient
    
    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        
        # Update base radius based on window size
        min_dimension = min(self.width(), self.height())
        self.base_radius = max(50, min_dimension // 4)
        self.max_radius = max(100, min_dimension // 2)
    
    def closeEvent(self, event):
        """Clean up resources when closing"""
        self.stop_audio_processing()
        if self.timer.isActive():
            self.timer.stop()
        self.audio.terminate()
        super().closeEvent(event)

class MainWindow(QMainWindow):
    """Main window for the visualization"""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Jarvis")
        self.setStyleSheet("background-color: black;")
        
        # Create and set central widget
        self.central_widget = PulsingLight()
        self.setCentralWidget(self.central_widget)
        
        # Set up window geometry
        self.setGeometry(100, 100, 800, 600)
        
        # Make window frameless and always on top
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint
        )

def launch_visualization():
    """Launch the visualization window"""
    try:
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        
        window = MainWindow()
        window.show()
        
        return app, window
    except Exception as e:
        logger.error(f"Failed to launch visualization: {e}")
        return None, None

# Create global visualization instance
visualization = None

def initialize():
    """Initialize the visualization"""
    global visualization
    app, window = launch_visualization()
    if app and window:
        visualization = window.central_widget
        return True
    return False

def cleanup():
    """Clean up visualization resources"""
    global visualization
    if visualization:
        visualization.close()
        visualization = None
