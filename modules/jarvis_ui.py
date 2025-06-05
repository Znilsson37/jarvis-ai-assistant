import sys
import os
import tkinter as tk
from tkinter import ttk, font
import threading
import time
import math
import random
from PIL import Image, ImageTk, ImageDraw, ImageFilter

# Ensure modules directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class JarvisUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JARVIS")
        self.root.configure(bg="#000B19")
        
        # Set fullscreen
        self.root.attributes("-fullscreen", True)
        
        # Allow escape key to exit fullscreen
        self.root.bind("<Escape>", self.exit_fullscreen)
        
        # Store the state
        self.listening = False
        self.speaking = False
        self.processing = False
        self.last_command = ""
        self.system_status = {}
        self.animation_frames = []
        self.current_frame = 0
        
        # Voice levels for animation
        self.voice_levels = [0] * 50
        
        # Create the UI elements
        self.setup_ui()
        
        # Start animation loop
        self.animate()
        
    def exit_fullscreen(self, event):
        self.root.attributes("-fullscreen", False)
        
    def setup_ui(self):
        # Create main frame
        self.main_frame = tk.Frame(self.root, bg="#000B19")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=2)
        self.main_frame.rowconfigure(0, weight=1)
        
        # Create left panel (status and visualizer)
        self.left_panel = tk.Frame(self.main_frame, bg="#000B19", padx=20, pady=20)
        self.left_panel.grid(row=0, column=0, sticky="nsew")
        
        # Create right panel (interaction and response)
        self.right_panel = tk.Frame(self.main_frame, bg="#000B19", padx=20, pady=20)
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        
        # Create canvas for animations in left panel
        self.canvas_height = 400
        self.canvas_width = 400
        self.canvas = tk.Canvas(self.left_panel, width=self.canvas_width, height=self.canvas_height, 
                               bg="#000B19", highlightthickness=0)
        self.canvas.pack(pady=20)
        
        # Create the circular visualizer
        self.create_circular_visualizer()
        
        # Status label
        self.status_label = tk.Label(self.left_panel, text="STANDBY", 
                                    font=("Orbitron", 16), bg="#000B19", fg="#0099FF")
        self.status_label.pack(pady=10)
        
        # System info section
        self.system_frame = tk.Frame(self.left_panel, bg="#001229", bd=1, relief=tk.GROOVE)
        self.system_frame.pack(fill=tk.X, pady=10)
        
        # CPU label
        self.cpu_frame = tk.Frame(self.system_frame, bg="#001229", pady=5, padx=10)
        self.cpu_frame.pack(fill=tk.X)
        self.cpu_label = tk.Label(self.cpu_frame, text="CPU:", font=("Orbitron", 10), 
                                 bg="#001229", fg="#66CCFF", anchor="w")
        self.cpu_label.pack(side=tk.LEFT)
        self.cpu_value = tk.Label(self.cpu_frame, text="0%", font=("Orbitron", 10), 
                                 bg="#001229", fg="#FFFFFF", anchor="e")
        self.cpu_value.pack(side=tk.RIGHT)
        
        # Memory label
        self.mem_frame = tk.Frame(self.system_frame, bg="#001229", pady=5, padx=10)
        self.mem_frame.pack(fill=tk.X)
        self.mem_label = tk.Label(self.mem_frame, text="MEMORY:", font=("Orbitron", 10), 
                                 bg="#001229", fg="#66CCFF", anchor="w")
        self.mem_label.pack(side=tk.LEFT)
        self.mem_value = tk.Label(self.mem_frame, text="0%", font=("Orbitron", 10), 
                                 bg="#001229", fg="#FFFFFF", anchor="e")
        self.mem_value.pack(side=tk.RIGHT)
        
        # Disk label
        self.disk_frame = tk.Frame(self.system_frame, bg="#001229", pady=5, padx=10)
        self.disk_frame.pack(fill=tk.X)
        self.disk_label = tk.Label(self.disk_frame, text="DISK:", font=("Orbitron", 10), 
                                  bg="#001229", fg="#66CCFF", anchor="w")
        self.disk_label.pack(side=tk.LEFT)
        self.disk_value = tk.Label(self.disk_frame, text="0%", font=("Orbitron", 10), 
                                  bg="#001229", fg="#FFFFFF", anchor="e")
        self.disk_value.pack(side=tk.RIGHT)
        
        # Right panel content
        # Logo
        self.logo_label = tk.Label(self.right_panel, text="J.A.R.V.I.S", 
                                  font=("Orbitron", 36, "bold"), bg="#000B19", fg="#0099FF")
        self.logo_label.pack(pady=20)
        
        # Subtitle
        self.subtitle = tk.Label(self.right_panel, text="Just A Rather Very Intelligent System", 
                               font=("Orbitron", 12), bg="#000B19", fg="#66CCFF")
        self.subtitle.pack()
        
        # Conversation frame
        self.conversation_frame = tk.Frame(self.right_panel, bg="#001229", bd=1, relief=tk.GROOVE)
        self.conversation_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Command history
        self.command_history = tk.Text(self.conversation_frame, bg="#001229", fg="#FFFFFF", 
                                      font=("Consolas", 12), wrap=tk.WORD, height=10,
                                      padx=10, pady=10, state=tk.DISABLED)
        self.command_history.pack(fill=tk.BOTH, expand=True)
        
        # Bottom controls
        self.controls_frame = tk.Frame(self.right_panel, bg="#000B19")
        self.controls_frame.pack(fill=tk.X, pady=10)
        
        # Current command label
        self.command_label = tk.Label(self.controls_frame, text="Waiting for command...", 
                                     font=("Orbitron", 14), bg="#000B19", fg="#0099FF",
                                     anchor="w")
        self.command_label.pack(fill=tk.X)
        
        # Progress bar for speech
        self.progress_style = ttk.Style()
        self.progress_style.theme_use('default')
        self.progress_style.configure("Horizontal.TProgressbar", 
                                     background="#0099FF", troughcolor="#001229")
        
        self.progress = ttk.Progressbar(self.controls_frame, style="Horizontal.TProgressbar", 
                                       orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, pady=10)
        
        # Control buttons
        self.button_frame = tk.Frame(self.controls_frame, bg="#000B19")
        self.button_frame.pack(fill=tk.X)
        
        # Manual listening button
        self.listen_button = tk.Button(self.button_frame, text="LISTEN", font=("Orbitron", 12),
                                     bg="#003366", fg="#FFFFFF", activebackground="#004080",
                                     command=self.toggle_listening)
        self.listen_button.pack(side=tk.LEFT, padx=5)
        
        # Stop button
        self.stop_button = tk.Button(self.button_frame, text="STOP", font=("Orbitron", 12),
                                   bg="#660000", fg="#FFFFFF", activebackground="#800000",
                                   command=self.stop_listening)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Exit button on the right
        self.exit_button = tk.Button(self.button_frame, text="EXIT", font=("Orbitron", 12),
                                   bg="#330033", fg="#FFFFFF", activebackground="#4D004D",
                                   command=self.root.destroy)
        self.exit_button.pack(side=tk.RIGHT, padx=5)
        
    def create_circular_visualizer(self):
        """Create the initial circular visualizer"""
        self.center_x = self.canvas_width // 2
        self.center_y = self.canvas_height // 2
        self.max_radius = min(self.center_x, self.center_y) - 10
        self.min_radius = self.max_radius // 2
        
        # Create the initial circle
        self.outer_circle = self.canvas.create_oval(
            self.center_x - self.max_radius, self.center_y - self.max_radius,
            self.center_x + self.max_radius, self.center_y + self.max_radius,
            outline="#0066CC", width=2, fill=""
        )
        
        self.inner_circle = self.canvas.create_oval(
            self.center_x - self.min_radius, self.center_y - self.min_radius,
            self.center_x + self.min_radius, self.center_y + self.min_radius,
            outline="#0099FF", width=2, fill=""
        )
        
        # Create lines
        self.lines = []
        num_lines = 36
        for i in range(num_lines):
            angle = 2 * math.pi * i / num_lines
            x1 = self.center_x + self.min_radius * math.cos(angle)
            y1 = self.center_y + self.min_radius * math.sin(angle)
            x2 = self.center_x + self.max_radius * math.cos(angle)
            y2 = self.center_y + self.max_radius * math.sin(angle)
            
            line = self.canvas.create_line(x1, y1, x2, y2, fill="#003366", width=1)
            self.lines.append(line)
        
        # Create voice level bars
        self.voice_bars = []
        for i in range(len(self.voice_levels)):
            angle = 2 * math.pi * i / len(self.voice_levels)
            x1 = self.center_x + (self.min_radius - 5) * math.cos(angle)
            y1 = self.center_y + (self.min_radius - 5) * math.sin(angle)
            x2 = self.center_x + (self.min_radius - 10) * math.cos(angle)
            y2 = self.center_y + (self.min_radius - 10) * math.sin(angle)
            
            bar = self.canvas.create_line(x1, y1, x2, y2, fill="#0099FF", width=3)
            self.voice_bars.append(bar)
            
    def update_voice_levels(self, levels=None):
        """Update the voice level visualization"""
        if levels:
            self.voice_levels = levels
        else:
            # Generate random voice levels for testing
            if self.listening:
                for i in range(len(self.voice_levels)):
                    # More variation when listening
                    self.voice_levels[i] = min(100, max(0, self.voice_levels[i] + 
                                                      random.randint(-15, 15)))
            else:
                # Less variation when idle
                for i in range(len(self.voice_levels)):
                    self.voice_levels[i] = min(30, max(0, self.voice_levels[i] + 
                                                     random.randint(-5, 5)))
        
        # Update the visual bars
        for i, level in enumerate(self.voice_levels):
            angle = 2 * math.pi * i / len(self.voice_levels)
            length = self.min_radius * level / 100
            
            x1 = self.center_x + (self.min_radius - 5) * math.cos(angle)
            y1 = self.center_y + (self.min_radius - 5) * math.sin(angle)
            x2 = self.center_x + (self.min_radius - 5 - length) * math.cos(angle)
            y2 = self.center_y + (self.min_radius - 5 - length) * math.sin(angle)
            
            self.canvas.coords(self.voice_bars[i], x1, y1, x2, y2)
            
            # Color based on intensity
            if level < 30:
                color = "#003366"  # Low level
            elif level < 70:
                color = "#0099FF"  # Medium level
            else:
                color = "#00FFFF"  # High level
                
            self.canvas.itemconfig(self.voice_bars[i], fill=color)
            
    def animate(self):
        """Animate the UI elements"""
        # Update voice levels
        self.update_voice_levels()
        
        # Rotate the outer circle slightly
        angle = 0.01 * (2 if self.processing else 1)
        self.rotate_lines(angle)
        
        # Pulse the inner circle if speaking
        if self.speaking:
            pulse = math.sin(time.time() * 5) * 10
            new_radius = self.min_radius + pulse
            self.canvas.coords(
                self.inner_circle,
                self.center_x - new_radius, self.center_y - new_radius,
                self.center_x + new_radius, self.center_y + new_radius
            )
        
        # Update progress if processing
        if self.processing:
            self.progress['value'] = (self.progress['value'] + 2) % 100
        
        # Schedule the next animation frame
        self.root.after(50, self.animate)
        
    def rotate_lines(self, angle):
        """Rotate the lines in the circular visualizer"""
        for i, line in enumerate(self.lines):
            coords = self.canvas.coords(line)
            
            # Rotate around center
            x1, y1, x2, y2 = coords
            
            # Translate to origin
            x1t, y1t = x1 - self.center_x, y1 - self.center_y
            x2t, y2t = x2 - self.center_x, y2 - self.center_y
            
            # Rotate
            x1r = x1t * math.cos(angle) - y1t * math.sin(angle)
            y1r = x1t * math.sin(angle) + y1t * math.cos(angle)
            x2r = x2t * math.cos(angle) - y2t * math.sin(angle)
            y2r = x2t * math.sin(angle) + y2t * math.cos(angle)
            
            # Translate back
            x1, y1 = x1r + self.center_x, y1r + self.center_y
            x2, y2 = x2r + self.center_x, y2r + self.center_y
            
            # Update line
            self.canvas.coords(line, x1, y1, x2, y2)
            
    def toggle_listening(self):
        """Toggle listening state"""
        self.listening = not self.listening
        
        if self.listening:
            self.status_label.config(text="LISTENING", fg="#00FF00")
            self.listen_button.config(text="PAUSE", bg="#004D00", activebackground="#006600")
            self.command_label.config(text="I'm listening...")
        else:
            self.status_label.config(text="STANDBY", fg="#0099FF")
            self.listen_button.config(text="LISTEN", bg="#003366", activebackground="#004080")
            self.command_label.config(text="Waiting for command...")
            
    def stop_listening(self):
        """Stop listening"""
        self.listening = False
        self.speaking = False
        self.processing = False
        self.status_label.config(text="STANDBY", fg="#0099FF")
        self.listen_button.config(text="LISTEN", bg="#003366", activebackground="#004080")
        self.command_label.config(text="Waiting for command...")
        self.progress['value'] = 0
        
    def set_processing(self, is_processing=True):
        """Set processing state"""
        self.processing = is_processing
        
        if is_processing:
            self.status_label.config(text="PROCESSING", fg="#FFCC00")
        else:
            if self.listening:
                self.status_label.config(text="LISTENING", fg="#00FF00")
            else:
                self.status_label.config(text="STANDBY", fg="#0099FF")
                
    def set_speaking(self, is_speaking=True):
        """Set speaking state"""
        self.speaking = is_speaking
        
        if is_speaking:
            self.status_label.config(text="SPEAKING", fg="#FF9900")
        else:
            if self.listening:
                self.status_label.config(text="LISTENING", fg="#00FF00")
            else:
                self.status_label.config(text="STANDBY", fg="#0099FF")
                
    def update_command(self, command):
        """Update the current command"""
        self.last_command = command
        self.command_label.config(text=f"{command}")
        
        # Add to history
        self.command_history.config(state=tk.NORMAL)
        self.command_history.insert(tk.END, f"You: {command}\n", "user")
        self.command_history.see(tk.END)
        self.command_history.config(state=tk.DISABLED)
        
    def update_response(self, response):
        """Update with Jarvis response"""
        # Add to history
        self.command_history.config(state=tk.NORMAL)
        self.command_history.insert(tk.END, f"JARVIS: {response}\n\n", "jarvis")
        self.command_history.see(tk.END)
        self.command_history.config(state=tk.DISABLED)
        
    def update_system_info(self, cpu=0, memory=0, disk=0):
        """Update system information"""
        self.cpu_value.config(text=f"{cpu}%")
        self.mem_value.config(text=f"{memory}%")
        self.disk_value.config(text=f"{disk}%")
        
        # Color coding based on values
        cpu_color = "#00FF00" if cpu < 70 else "#FFCC00" if cpu < 90 else "#FF0000"
        mem_color = "#00FF00" if memory < 70 else "#FFCC00" if memory < 90 else "#FF0000"
        disk_color = "#00FF00" if disk < 70 else "#FFCC00" if disk < 90 else "#FF0000"
        
        self.cpu_value.config(fg=cpu_color)
        self.mem_value.config(fg=mem_color)
        self.disk_value.config(fg=disk_color)


# Class to integrate with existing Jarvis assistant
class JarvisUIIntegration:
    def __init__(self):
        self.ui = None
        self.root = None
        
    def start_ui(self):
        """Start the UI in a separate thread"""
        # Create and configure the UI thread
        self.root = tk.Tk()
        
        # Try to load a futuristic font if available
        self.load_fonts()
        
        self.ui = JarvisUI(self.root)
        self.root.mainloop()
        
    def load_fonts(self):
        """Load custom fonts if available"""
        try:
            # Try common futuristic fonts
            available_fonts = font.families()
            preferred_fonts = ["Orbitron", "Exo", "Roboto", "Century Gothic", "Verdana"]
            
            for font_name in preferred_fonts:
                if font_name in available_fonts:
                    default_font = font.nametofont("TkDefaultFont")
                    default_font.config(family=font_name)
                    break
        except:
            pass  # Use default fonts if custom fonts are not available
    
    def on_listening(self):
        """Called when Jarvis starts listening"""
        if self.ui:
            self.ui.toggle_listening()
    
    def on_processing_command(self, command):
        """Called when processing a command"""
        if self.ui:
            self.ui.set_processing(True)
            self.ui.update_command(command)
    
    def on_speaking(self, response):
        """Called when Jarvis speaks"""
        if self.ui:
            self.ui.set_speaking(True)
            self.ui.update_response(response)
    
    def on_finish_speaking(self):
        """Called when Jarvis finishes speaking"""
        if self.ui:
            self.ui.set_speaking(False)
            self.ui.set_processing(False)
    
    def update_system_info(self, cpu, memory, disk):
        """Update system info in the UI"""
        if self.ui:
            self.ui.update_system_info(cpu, memory, disk)


# Function to start UI in standalone mode (for testing)
def start_ui_standalone():
    root = tk.Tk()
    app = JarvisUI(root)
    
    # Demo functionality for testing
    def demo_functions():
        import random
        import time
        
        # Simulate listening
        app.toggle_listening()
        time.sleep(2)
        
        # Simulate processing command
        app.set_processing(True)
        app.update_command("Jarvis, what's the system status?")
        time.sleep(1.5)
        
        # Simulate response
        app.set_processing(False)
        app.set_speaking(True)
        app.update_response("System status: CPU usage is 34%, Memory usage is 62%, Disk usage is 45%")
        app.update_system_info(34, 62, 45)
        time.sleep(3)
        
        # Simulate end of speaking
        app.set_speaking(False)
        time.sleep(1)
        
        # Another command
        app.set_processing(True)
        app.update_command("Jarvis, open Chrome")
        time.sleep(1)
        
        app.set_processing(False)
        app.set_speaking(True)
        app.update_response("Opening Chrome for you.")
        time.sleep(2)
        
        app.set_speaking(False)
        app.toggle_listening()
    
    # Run demo in separate thread
    import threading
    demo_thread = threading.Thread(target=demo_functions)
    demo_thread.daemon = True
    demo_thread.start()
    
    root.mainloop()


# Run standalone UI for testing if executed directly
if __name__ == "__main__":
    start_ui_standalone()