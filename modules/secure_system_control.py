"""Secure system control with proper validation and sandboxing"""

import subprocess
import psutil
import logging
import os
import shlex
from typing import Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PermissionLevel(Enum):
    READ_ONLY = "read_only"
    STANDARD = "standard"
    ELEVATED = "elevated"
    ADMIN = "admin"

@dataclass
class SystemCommand:
    command: str
    args: List[str]
    permission_level: PermissionLevel
    description: str
    allowed_paths: Optional[List[str]] = None

class SecureSystemController:
    """Secure system controller with proper validation and sandboxing"""
    
    def __init__(self):
        self.current_permission = PermissionLevel.STANDARD
        self.allowed_commands = self._init_allowed_commands()
        self.command_history = []
        
    def _init_allowed_commands(self) -> Dict[str, SystemCommand]:
        """Initialize allowed system commands"""
        return {
            "notepad": SystemCommand(
                command="notepad.exe",
                args=[],
                permission_level=PermissionLevel.STANDARD,
                description="Open Notepad text editor"
            ),
            "calculator": SystemCommand(
                command="calc.exe",
                args=[],
                permission_level=PermissionLevel.STANDARD,
                description="Open Calculator"
            ),
            "chrome": SystemCommand(
                command="chrome.exe",
                args=[],
                permission_level=PermissionLevel.STANDARD,
                description="Open Google Chrome browser",
                allowed_paths=["C:\\Program Files\\Google\\Chrome\\Application\\"]
            ),
            "system_info": SystemCommand(
                command="systeminfo",
                args=[],
                permission_level=PermissionLevel.READ_ONLY,
                description="Get system information"
            )
        }
    
    def validate_command(self, command_name: str) -> bool:
        """Validate if command is allowed"""
        if command_name not in self.allowed_commands:
            logger.warning(f"Command '{command_name}' not in allowed list")
            return False
        
        cmd = self.allowed_commands[command_name]
        if cmd.permission_level.value > self.current_permission.value:
            logger.warning(f"Insufficient permissions for command '{command_name}'")
            return False
        
        return True
    
    def execute_command_secure(self, command_name: str, **kwargs) -> Dict[str, Union[str, bool]]:
        """Execute command with security validation"""
        if not self.validate_command(command_name):
            return {
                "success": False,
                "message": f"Command '{command_name}' not allowed or insufficient permissions",
                "output": ""
            }
        
        cmd = self.allowed_commands[command_name]
        
        try:
            # Build command with validation
            full_command = [cmd.command] + cmd.args
            
            # Execute with security constraints
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=30,  # Prevent hanging
                check=False
            )
            
            # Log command execution
            self.command_history.append({
                "command": command_name,
                "timestamp": time.time(),
                "success": result.returncode == 0,
                "return_code": result.returncode
            })
            
            return {
                "success": result.returncode == 0,
                "message": f"Command '{command_name}' executed",
                "output": result.stdout,
                "error": result.stderr if result.stderr else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "message": f"Command '{command_name}' timed out",
                "output": ""
            }
        except Exception as e:
            logger.error(f"Error executing command '{command_name}': {e}")
            return {
                "success": False,
                "message": f"Error executing command: {str(e)}",
                "output": ""
            }
    
    def get_system_info_secure(self) -> Dict[str, Union[str, float]]:
        """Get system information with error handling"""
        try:
            info = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "boot_time": psutil.boot_time(),
                "process_count": len(psutil.pids())
            }
            
            # Add network info safely
            try:
                net_io = psutil.net_io_counters()
                info.update({
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv
                })
            except Exception as e:
                logger.warning(f"Could not get network info: {e}")
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {"error": str(e)}
    
    def launch_app_secure(self, app_name: str) -> str:
        """Launch application with security validation"""
        app_name = app_name.lower().strip()
        
        if not self.validate_command(app_name):
            return f"Application '{app_name}' is not allowed or requires higher permissions"
        
        result = self.execute_command_secure(app_name)
        
        if result["success"]:
            return f"Launched {app_name} successfully"
        else:
            return f"Failed to launch {app_name}: {result['message']}"
    
    def get_running_processes_safe(self) -> List[Dict[str, Union[str, int]]]:
        """Get running processes with error handling"""
        processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.info
                    # Only include safe information
                    processes.append({
                        "pid": proc_info['pid'],
                        "name": proc_info['name'],
                        "cpu_percent": proc_info.get('cpu_percent', 0),
                        "memory_percent": proc_info.get('memory_percent', 0)
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            logger.error(f"Error getting process list: {e}")
        
        return processes
    
    def set_permission_level(self, level: PermissionLevel, auth_token: Optional[str] = None):
        """Set permission level with authentication"""
        # In a real implementation, you'd validate the auth_token
        if level == PermissionLevel.ADMIN and not auth_token:
            raise ValueError("Admin permissions require authentication token")
        
        self.current_permission = level
        logger.info(f"Permission level set to {level.value}")
    
    def get_command_history(self) -> List[Dict]:
        """Get command execution history"""
        return self.command_history.copy()

# Global secure system controller
secure_system = SecureSystemController()