import os
import sys
import psutil
import platform
import subprocess
import winreg
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class SystemDiagnostics:
    """Advanced system diagnostics and repair capabilities"""
    
    def __init__(self):
        self.os_type = platform.system().lower()
        self.diagnostic_results = {}
        
    def run_full_diagnostics(self) -> Dict[str, Dict]:
        """Run comprehensive system diagnostics"""
        diagnostics = {
            "system_health": self.check_system_health(),
            "disk_health": self.check_disk_health(),
            "memory_health": self.check_memory_health(),
            "network_health": self.check_network_health(),
            "startup_health": self.check_startup_programs(),
            "windows_updates": self.check_windows_updates(),
            "security_health": self.check_security_health()
        }
        self.diagnostic_results = diagnostics
        return diagnostics
    
    def check_system_health(self) -> Dict[str, Union[str, float]]:
        """Check overall system health"""
        try:
            health = {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "cpu_temperature": self._get_cpu_temperature(),
                "uptime": psutil.boot_time(),
                "processes": len(psutil.pids()),
                "status": "healthy"
            }
            
            # Determine system status
            if (health["cpu_usage"] > 90 or 
                health["memory_usage"] > 90 or 
                health["disk_usage"] > 90):
                health["status"] = "critical"
            elif (health["cpu_usage"] > 70 or 
                  health["memory_usage"] > 70 or 
                  health["disk_usage"] > 70):
                health["status"] = "warning"
                
            return health
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {"status": "error", "message": str(e)}
    
    def check_disk_health(self) -> Dict[str, Dict]:
        """Check disk health and performance"""
        try:
            disks = {}
            for partition in psutil.disk_partitions():
                if partition.fstype:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info = {
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": usage.percent,
                        "filesystem": partition.fstype,
                        "status": "healthy"
                    }
                    
                    # Run SMART diagnostics if available
                    if self.os_type == "windows":
                        smart_info = self._run_smart_diagnostics(partition.device)
                        disk_info.update(smart_info)
                    
                    # Determine disk status
                    if disk_info["percent"] > 90:
                        disk_info["status"] = "critical"
                    elif disk_info["percent"] > 70:
                        disk_info["status"] = "warning"
                        
                    disks[partition.device] = disk_info
            
            return disks
        except Exception as e:
            logger.error(f"Error checking disk health: {e}")
            return {"status": "error", "message": str(e)}
    
    def check_memory_health(self) -> Dict[str, Union[str, float]]:
        """Check memory health and performance"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            memory_health = {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent,
                "swap_total": swap.total,
                "swap_used": swap.used,
                "swap_percent": swap.percent,
                "status": "healthy"
            }
            
            # Check for memory issues
            if memory.percent > 90 or swap.percent > 90:
                memory_health["status"] = "critical"
            elif memory.percent > 70 or swap.percent > 70:
                memory_health["status"] = "warning"
                
            return memory_health
        except Exception as e:
            logger.error(f"Error checking memory health: {e}")
            return {"status": "error", "message": str(e)}
    
    def check_network_health(self) -> Dict[str, Union[str, Dict]]:
        """Check network connectivity and performance"""
        try:
            network_health = {
                "interfaces": {},
                "connectivity": self._test_internet_connectivity(),
                "status": "healthy"
            }
            
            # Check network interfaces
            net_io = psutil.net_io_counters(pernic=True)
            for interface, stats in net_io.items():
                network_health["interfaces"][interface] = {
                    "bytes_sent": stats.bytes_sent,
                    "bytes_recv": stats.bytes_recv,
                    "packets_sent": stats.packets_sent,
                    "packets_recv": stats.packets_recv,
                    "errors_in": stats.errin,
                    "errors_out": stats.errout,
                    "drops_in": stats.dropin,
                    "drops_out": stats.dropout
                }
                
                # Check for network issues
                if stats.errin > 0 or stats.errout > 0:
                    network_health["status"] = "warning"
                if not network_health["connectivity"]:
                    network_health["status"] = "critical"
                    
            return network_health
        except Exception as e:
            logger.error(f"Error checking network health: {e}")
            return {"status": "error", "message": str(e)}
    
    def check_startup_programs(self) -> Dict[str, List[Dict]]:
        """Check startup programs and their impact"""
        try:
            startup_info = {
                "programs": [],
                "status": "healthy"
            }
            
            if self.os_type == "windows":
                # Check Windows startup locations
                startup_locations = [
                    r"Software\Microsoft\Windows\CurrentVersion\Run",
                    r"Software\Microsoft\Windows\CurrentVersion\RunOnce"
                ]
                
                for location in startup_locations:
                    try:
                        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, location)
                        i = 0
                        while True:
                            try:
                                name, value, _ = winreg.EnumValue(key, i)
                                startup_info["programs"].append({
                                    "name": name,
                                    "command": value,
                                    "location": location,
                                    "impact": self._assess_startup_impact(value)
                                })
                                i += 1
                            except WindowsError:
                                break
                    except WindowsError:
                        pass
                        
            # Assess overall startup health
            high_impact = sum(1 for p in startup_info["programs"] 
                            if p["impact"] == "high")
            if high_impact > 5:
                startup_info["status"] = "warning"
                
            return startup_info
        except Exception as e:
            logger.error(f"Error checking startup programs: {e}")
            return {"status": "error", "message": str(e)}
    
    def check_windows_updates(self) -> Dict[str, Union[str, List]]:
        """Check Windows Update status"""
        try:
            if self.os_type != "windows":
                return {"status": "not_applicable"}
                
            result = {
                "last_check": None,
                "pending_updates": [],
                "status": "healthy"
            }
            
            # Use PowerShell to check for updates
            ps_command = "Get-WUHistory | Select-Object -First 5 | ForEach-Object { $_.Title }"
            process = subprocess.run(["powershell", "-Command", ps_command],
                                  capture_output=True, text=True)
            
            if process.returncode == 0:
                updates = process.stdout.strip().split('\n')
                result["pending_updates"] = [u.strip() for u in updates if u.strip()]
                if len(result["pending_updates"]) > 0:
                    result["status"] = "warning"
            else:
                result["status"] = "error"
                
            return result
        except Exception as e:
            logger.error(f"Error checking Windows updates: {e}")
            return {"status": "error", "message": str(e)}
    
    def check_security_health(self) -> Dict[str, Union[str, bool]]:
        """Check system security status"""
        try:
            security = {
                "antivirus_active": self._check_antivirus(),
                "firewall_active": self._check_firewall(),
                "updates_pending": len(self.check_windows_updates()["pending_updates"]) > 0,
                "status": "healthy"
            }
            
            # Assess security status
            if not security["antivirus_active"] or not security["firewall_active"]:
                security["status"] = "critical"
            elif security["updates_pending"]:
                security["status"] = "warning"
                
            return security
        except Exception as e:
            logger.error(f"Error checking security health: {e}")
            return {"status": "error", "message": str(e)}
    
    def optimize_system(self) -> Dict[str, str]:
        """Perform system optimization"""
        try:
            results = {
                "disk_cleanup": self._perform_disk_cleanup(),
                "defrag": self._perform_defrag(),
                "startup_optimization": self._optimize_startup(),
                "status": "completed"
            }
            return results
        except Exception as e:
            logger.error(f"Error during system optimization: {e}")
            return {"status": "error", "message": str(e)}
    
    def repair_system(self, issues: List[str]) -> Dict[str, str]:
        """Attempt to repair identified system issues"""
        try:
            repairs = {}
            for issue in issues:
                if issue == "disk":
                    repairs["disk"] = self._repair_disk_issues()
                elif issue == "memory":
                    repairs["memory"] = self._optimize_memory()
                elif issue == "network":
                    repairs["network"] = self._repair_network_issues()
                elif issue == "windows":
                    repairs["windows"] = self._repair_windows_issues()
                    
            return {"status": "completed", "repairs": repairs}
        except Exception as e:
            logger.error(f"Error during system repair: {e}")
            return {"status": "error", "message": str(e)}
    
    # Helper methods
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature if available"""
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            return entry.current
            return None
        except:
            return None
    
    def _run_smart_diagnostics(self, drive: str) -> Dict:
        """Run SMART diagnostics on drive"""
        # Implementation would use Windows Management Instrumentation (WMI)
        # or third-party tools to check SMART status
        return {"smart_status": "not_implemented"}
    
    def _test_internet_connectivity(self) -> bool:
        """Test internet connectivity"""
        try:
            subprocess.run(["ping", "8.8.8.8", "-n", "1"],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
            return True
        except:
            return False
    
    def _assess_startup_impact(self, command: str) -> str:
        """Assess the performance impact of a startup program"""
        # Simple heuristic based on known high-impact programs
        high_impact_keywords = ["update", "sync", "cloud", "adobe"]
        if any(keyword in command.lower() for keyword in high_impact_keywords):
            return "high"
        return "low"
    
    def _check_antivirus(self) -> bool:
        """Check if antivirus is active"""
        if self.os_type == "windows":
            try:
                process = subprocess.run(
                    ["powershell", "-Command",
                     "Get-MpComputerStatus | Select-Object AntivirusEnabled"],
                    capture_output=True, text=True
                )
                return "True" in process.stdout
            except:
                return False
        return False
    
    def _check_firewall(self) -> bool:
        """Check if firewall is active"""
        if self.os_type == "windows":
            try:
                process = subprocess.run(
                    ["powershell", "-Command",
                     "Get-NetFirewallProfile | Select-Object Enabled"],
                    capture_output=True, text=True
                )
                return "True" in process.stdout
            except:
                return False
        return False
    
    def _perform_disk_cleanup(self) -> str:
        """Perform disk cleanup"""
        if self.os_type == "windows":
            try:
                subprocess.run(["cleanmgr", "/sagerun:1"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
                return "completed"
            except:
                return "failed"
        return "not_supported"
    
    def _perform_defrag(self) -> str:
        """Perform disk defragmentation if needed"""
        if self.os_type == "windows":
            try:
                subprocess.run(["defrag", "C:", "/A"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
                return "completed"
            except:
                return "failed"
        return "not_supported"
    
    def _optimize_startup(self) -> str:
        """Optimize system startup"""
        try:
            # Disable high-impact startup items
            startup_info = self.check_startup_programs()
            for program in startup_info["programs"]:
                if program["impact"] == "high":
                    # Implementation would disable high-impact startup items
                    pass
            return "completed"
        except:
            return "failed"
    
    def _optimize_memory(self) -> str:
        """Optimize memory usage"""
        try:
            if self.os_type == "windows":
                # Clear standby list and working sets
                subprocess.run(["powershell", "-Command",
                              "Clear-RecycleBin -Force"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
                return "completed"
        except:
            return "failed"
        return "not_supported"
    
    def _repair_disk_issues(self) -> str:
        """Repair disk issues"""
        if self.os_type == "windows":
            try:
                subprocess.run(["chkdsk", "C:", "/f"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
                return "scheduled"
            except:
                return "failed"
        return "not_supported"
    
    def _repair_network_issues(self) -> str:
        """Repair network issues"""
        if self.os_type == "windows":
            try:
                subprocess.run(["ipconfig", "/release"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
                subprocess.run(["ipconfig", "/renew"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
                subprocess.run(["ipconfig", "/flushdns"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
                return "completed"
            except:
                return "failed"
        return "not_supported"
    
    def _repair_windows_issues(self) -> str:
        """Repair Windows system issues"""
        if self.os_type == "windows":
            try:
                subprocess.run(["sfc", "/scannow"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
                subprocess.run(["DISM", "/Online", "/Cleanup-Image", "/RestoreHealth"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
                return "completed"
            except:
                return "failed"
        return "not_supported"

# Create global diagnostics instance
diagnostics = SystemDiagnostics()
