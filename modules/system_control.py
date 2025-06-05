import os
import subprocess
import shutil
import psutil
import ctypes
import platform
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

class SystemController:
    def __init__(self):
        self.os_type = platform.system().lower()
        self.known_apps = self._load_known_apps()

    def _load_known_apps(self) -> Dict[str, str]:
        """Load known application paths based on OS"""
        if self.os_type == "windows":
            return {
                "chrome": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                "firefox": "C:\\Program Files\\Mozilla Firefox\\firefox.exe",
                "edge": "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
                "notepad": "notepad.exe",
                "calculator": "calc.exe",
                "word": "WINWORD.EXE",
                "excel": "EXCEL.EXE",
                "powerpoint": "POWERPNT.EXE",
                "explorer": "explorer.exe",
                "cmd": "cmd.exe",
                "control": "control.exe",
                "task_manager": "taskmgr.exe",
                "zoom": "C:\\Users\\Administrator\\AppData\\Roaming\\Zoom\\bin\\Zoom.exe"
            }
        else:
            return {}  # Add paths for other OS types as needed

    def get_time(self) -> str:
        """Get current time"""
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}"

    def launch_app(self, app_name: str) -> str:
        """Launch application by name"""
        app_name = app_name.lower()
        try:
            if app_name in self.known_apps:
                subprocess.Popen(self.known_apps[app_name])
                return f"Launching {app_name}"
            else:
                # Try to launch directly if it's an executable name
                subprocess.Popen(app_name)
                return f"Attempting to launch {app_name}"
        except Exception as e:
            return f"Failed to launch {app_name}: {str(e)}"

    def close_app(self, app_name: str) -> str:
        """Close application by name"""
        try:
            os.system(f"taskkill /f /im {app_name}.exe")
            return f"Closed {app_name}"
        except Exception as e:
            return f"Failed to close {app_name}: {str(e)}"

    def get_running_apps(self) -> List[str]:
        """Get list of running applications"""
        running_apps = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                running_apps.append(proc.info['name'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return list(set(running_apps))

    def organize_files(self, directory: Union[str, Path]) -> str:
        """Organize files in directory by extension"""
        directory = Path(directory)
        if not directory.exists():
            return f"Directory {directory} does not exist."

        try:
            for file in directory.iterdir():
                if file.is_file():
                    ext = file.suffix.lower().strip(".") or "misc"
                    target_dir = directory / ext
                    target_dir.mkdir(exist_ok=True)
                    shutil.move(str(file), str(target_dir / file.name))
            return f"Organized files in {directory}"
        except Exception as e:
            return f"Error organizing files: {str(e)}"

    def get_system_info(self) -> Dict[str, Union[str, float]]:
        """Get comprehensive system information"""
        try:
            info = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "battery": self.get_battery_info(),
                "network": self.get_network_info(),
                "os": platform.system(),
                "os_version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            }
            return info
        except Exception as e:
            return {"error": str(e)}

    def get_battery_info(self) -> Dict[str, Union[float, bool, str]]:
        """Get battery status information"""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return {
                    "percent": battery.percent,
                    "power_plugged": battery.power_plugged,
                    "time_left": str(datetime.timedelta(seconds=battery.secsleft)) if battery.secsleft > 0 else "Unknown"
                }
            return {"error": "No battery detected"}
        except Exception as e:
            return {"error": str(e)}

    def get_network_info(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """Get network interface information"""
        try:
            net_io = psutil.net_io_counters(pernic=True)
            return {iface: {
                "bytes_sent": stats.bytes_sent,
                "bytes_recv": stats.bytes_recv,
                "packets_sent": stats.packets_sent,
                "packets_recv": stats.packets_recv
            } for iface, stats in net_io.items()}
        except Exception as e:
            return {"error": str(e)}

    def set_volume(self, level: int) -> str:
        """Set system volume level (0-100)"""
        try:
            if self.os_type == "windows":
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                
                # Convert to volume range (0.0 to 1.0)
                volume_level = max(0.0, min(1.0, level / 100.0))
                volume.SetMasterVolumeLevelScalar(volume_level, None)
                return f"Volume set to {level}%"
        except Exception as e:
            return f"Failed to set volume: {str(e)}"

    def set_display_brightness(self, level: int) -> str:
        """Set display brightness (0-100)"""
        try:
            if self.os_type == "windows":
                import wmi
                c = wmi.WMI(namespace='wmi')
                methods = c.WmiMonitorBrightnessMethods()[0]
                methods.WmiSetBrightness(level, 0)
                return f"Brightness set to {level}%"
        except Exception as e:
            return f"Failed to set brightness: {str(e)}"

    def toggle_wifi(self, enable: bool) -> str:
        """Toggle WiFi on/off"""
        try:
            if self.os_type == "windows":
                action = "enable" if enable else "disable"
                os.system(f"netsh interface set interface 'Wi-Fi' admin={action}")
                return f"WiFi {'enabled' if enable else 'disabled'}"
        except Exception as e:
            return f"Failed to toggle WiFi: {str(e)}"

    def set_power_mode(self, mode: str) -> str:
        """Set power mode (balanced, power_saver, high_performance)"""
        try:
            if self.os_type == "windows":
                modes = {
                    "balanced": "balanced",
                    "power_saver": "powersaver",
                    "high_performance": "high"
                }
                if mode in modes:
                    os.system(f"powercfg /setactive scheme_{modes[mode]}")
                    return f"Power mode set to {mode}"
                return "Invalid power mode"
        except Exception as e:
            return f"Failed to set power mode: {str(e)}"

    def lock_system(self) -> str:
        """Lock the system"""
        try:
            if self.os_type == "windows":
                ctypes.windll.user32.LockWorkStation()
                return "System locked"
        except Exception as e:
            return f"Failed to lock system: {str(e)}"

    def shutdown_system(self, restart: bool = False) -> str:
        """Shutdown or restart the system"""
        try:
            if restart:
                if self.os_type == "windows":
                    os.system("shutdown /r /t 0")
                else:
                    os.system("shutdown -r now")
                return "Restarting system..."
            else:
                if self.os_type == "windows":
                    os.system("shutdown /s /t 0")
                else:
                    os.system("shutdown -h now")
                return "Shutting down system..."
        except Exception as e:
            return f"Failed to {'restart' if restart else 'shutdown'} system: {str(e)}"

# Create global system controller instance
system = SystemController()
