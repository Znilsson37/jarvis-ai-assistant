import sys
import os
import tkinter as tk
from tkinter import ttk, font
import threading
import time
import math
import random
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import numpy as np

# Ensure modules directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class PulsingLight:
    def __init__(self, canvas, x, y, color="#FFFFFF"):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.base_color = color
        self.radius = 0
        self.max_radius = 300
        self.opacity = 255
        self.rings = []
        self.intensity = 1.0
        self.particles = []
        
    def create_particle(self):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 3)
        return {
            'x': self.x,
            'y': self.y,
            'dx': math.cos(angle) * speed,
            'dy': math.sin(angle) * speed,
            'life': 1.0,
            'size': random.uniform(2, 4)
        }
        
    def update(self, audio_level=0.5):
        # Update intensity based on audio level
        target_intensity = 0.5 + audio_level * 2
        self.intensity += (target_intensity - self.intensity) * 0.1
        
        # Create new rings periodically
        if len(self.rings) < 5 and random.random() < 0.1:
            self.rings.append({
                'radius': 0,
                'opacity': 255,
                'speed': random.uniform(3, 6),
                'thickness': random.uniform(1, 3)
            })
        
        # Create particles
        if random.random() < self.intensity * 0.3:
            self.particles.append(self.create_particle())
        
        # Update existing rings
        new_rings = []
        for ring in self.rings:
            ring['radius'] += ring['speed'] * self.intensity
            ring['opacity'] = max(0, 255 * (1 - ring['radius'] / self.max_radius))
            if ring['opacity'] > 0:
                new_rings.append(ring)
        self.rings = new_rings
        
        # Update particles
        new_particles = []
        for particle in self.particles:
            particle['x'] += particle['dx']
            particle['y'] += particle['dy']
            particle['life'] -= 0.02
            if particle['life'] > 0:
                new_particles.append(particle)
        self.particles = new_particles
        
        # Draw everything
        self.canvas.delete("pulse")
        
        # Draw particles
        for particle in self.particles:
            opacity = int(255 * particle['life'])
            color = f"#{opacity:02x}{opacity:02x}{opacity:02x}"
            size = particle['size'] * particle['life']
            self.canvas.create_oval(
                particle['x'] - size, particle['y'] - size,
                particle['x'] + size, particle['y'] + size,
                fill=color,
                outline="",
                tags="pulse"
            )
        
        # Draw rings
        for ring in self.rings:
            # Create main ring
            x0 = self.x - ring['radius']
            y0 = self.y - ring['radius']
            x1 = self.x + ring['radius']
            y1 = self.y + ring['radius']
            
            # Create gradient effect
            for i in range(3):
                offset = i * 2
                opacity = int(ring['opacity'] * (1 - i * 0.3))
                color = f"#{opacity:02x}{opacity:02x}{opacity:02x}"
                width = ring['thickness'] * (1 - i * 0.2)
                self.canvas.create_oval(
                    x0 - offset, y0 - offset,
                    x1 + offset, y1 + offset,
                    outline=color,
                    width=width,
                    tags="pulse"
                )
            
            # Add inner glow
            inner_radius = max(0, ring['radius'] - 10)
            if inner_radius > 0:
                x0_inner = self.x - inner_radius
                y0_inner = self.y - inner_radius
                x1_inner = self.x + inner_radius
                y1_inner = self.y + inner_radius
                inner_opacity = int(ring['opacity'] * 0.5)
                inner_color = f"#{inner_opacity:02x}{inner_opacity:02x}{inner_opacity:02x}"
                self.canvas.create_oval(
                    x0_inner, y0_inner,
                    x1_inner, y1_inner,
                    outline=inner_color,
                    width=1,
                    tags="pulse"
                )
        
        # Draw center glow
        center_size = 20 * self.intensity
        center_opacity = int(200 * self.intensity)
        center_color = f"#{center_opacity:02x}{center_opacity:02x}{center_opacity:02x}"
        self.canvas.create_oval(
            self.x - center_size, self.y - center_size,
            self.x + center_size, self.y + center_size,
            fill=center_color,
            outline="",
            tags="pulse"
        )

class JarvisUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JARVIS")
        self.root.configure(bg="#000000")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", self.exit_fullscreen)

        self.listening = False
        self.speaking = False
        self.processing = False
        
        # Create main canvas
        self.canvas = tk.Canvas(
            self.root,
            bg="#000000",
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initialize center coordinates
        self.center_x = self.root.winfo_screenwidth() // 2
        self.center_y = self.root.winfo_screenheight() // 2
        
        # Create pulsing light effect
        self.light = PulsingLight(self.canvas, self.center_x, self.center_y)
        
        # Create status display with a more futuristic look
        self.status_frame = tk.Frame(self.root, bg="#000000")
        self.status_frame.place(relx=0.5, rely=0.9, anchor="center")
        
        self.status_label = tk.Label(
            self.status_frame,
            text="STANDBY",
            font=("Orbitron", 24),
            bg="#000000",
            fg="#CCCCCC",
            padx=20,
            pady=10
        )
        self.status_label.pack()
        
        # Add decorative lines
        line_length = 100
        line_color = "#333333"
        self.canvas.create_line(
            self.center_x - line_length, self.center_y + 200,
            self.center_x + line_length, self.center_y + 200,
            fill=line_color,
            width=2
        )
        
        # Initialize audio simulation
        self.audio_level = 0.0
        self.target_audio = 0.0
        
        # Start animation
        self.animate()

    def exit_fullscreen(self, event):
        self.root.attributes("-fullscreen", False)

    def toggle_listening(self):
        self.listening = not self.listening
        if self.listening:
            self.target_audio = 0.7
        else:
            self.target_audio = 0.3
        self.update_status()

    def set_speaking(self, is_speaking=True):
        self.speaking = is_speaking
        if is_speaking:
            self.target_audio = 0.8
        else:
            self.target_audio = 0.3
        self.update_status()

    def set_processing(self, is_processing=True):
        self.processing = is_processing
        if is_processing:
            self.target_audio = 0.5
        else:
            self.target_audio = 0.3
        self.update_status()

    def update_status(self):
        if self.processing:
            self.status_label.config(text="PROCESSING", fg="#FFCC00")
        elif self.speaking:
            self.status_label.config(text="SPEAKING", fg="#FF9900")
        elif self.listening:
            self.status_label.config(text="LISTENING", fg="#00FF00")
        else:
            self.status_label.config(text="STANDBY", fg="#CCCCCC")

    def update_command(self, command):
        pass

    def update_response(self, response):
        pass

    def update_system_info(self, cpu, memory, disk):
        pass

    def animate(self):
        # Update simulated audio level with smooth transitions
        self.audio_level += (self.target_audio - self.audio_level) * 0.1
        
        # Add some random variation for more natural movement
        variation = random.uniform(-0.1, 0.1)
        current_level = max(0, min(1, self.audio_level + variation))
        
        # Update pulsing light
        self.light.update(current_level)
        
        # Schedule next frame
        self.root.after(20, self.animate)

class JarvisUIIntegration:
    def __init__(self):
        self.ui = None
        self.root = None

    def start_ui(self):
        self.root = tk.Tk()
        self.load_fonts()
        self.ui = JarvisUI(self.root)
        self.root.mainloop()

    def load_fonts(self):
        try:
            available_fonts = font.families()
            preferred_fonts = ["Orbitron", "Exo", "Roboto", "Century Gothic", "Verdana"]
            for font_name in preferred_fonts:
                if font_name in available_fonts:
                    default_font = font.nametofont("TkDefaultFont")
                    default_font.config(family=font_name)
                    break
        except:
            pass

    def on_listening(self):
        if self.ui:
            self.root.after(0, self.ui.toggle_listening)

    def on_processing_command(self, command):
        if self.ui:
            self.root.after(0, lambda: self.ui.set_processing(True))
            self.root.after(0, lambda: self.ui.update_command(command))

    def on_speaking(self, response):
        if self.ui:
            self.root.after(0, lambda: self.ui.set_speaking(True))
            self.root.after(0, lambda: self.ui.update_response(response))

    def on_finish_speaking(self):
        if self.ui:
            self.root.after(0, lambda: self.ui.set_speaking(False))
            self.root.after(0, lambda: self.ui.set_processing(False))

    def update_system_info(self, cpu, memory, disk):
        if self.ui:
            self.root.after(0, lambda: self.ui.update_system_info(cpu, memory, disk))

def start_ui_standalone():
    root = tk.Tk()
    app = JarvisUI(root)

    def demo_functions():
        import time
        while True:
            app.toggle_listening()
            time.sleep(2)
            app.set_processing(True)
            app.update_command("Jarvis, what's the system status?")
            time.sleep(1.5)
            app.set_processing(False)
            app.set_speaking(True)
            app.update_response("System status: All systems operational")
            time.sleep(3)
            app.set_speaking(False)
            time.sleep(1)
            app.toggle_listening()
            time.sleep(2)

    demo_thread = threading.Thread(target=demo_functions)
    demo_thread.daemon = True
    demo_thread.start()

    root.mainloop()

if __name__ == "__main__":
    start_ui_standalone()
