"""Comprehensive monitoring and health checking system"""

import time
import threading
import logging
import psutil
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import queue

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthMetric:
    name: str
    value: float
    status: HealthStatus
    threshold_warning: float
    threshold_critical: float
    timestamp: datetime
    unit: str = ""
    description: str = ""

@dataclass
class SystemAlert:
    id: str
    severity: str
    message: str
    component: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class HealthMonitor:
    """Comprehensive health monitoring system"""
    
    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.alert_callbacks = []
        self.monitoring_thread = None
        self.is_monitoring = False
        self.check_interval = 30  # seconds
        
        # Thresholds
        self.thresholds = {
            "cpu_percent": {"warning": 70, "critical": 90},
            "memory_percent": {"warning": 80, "critical": 95},
            "disk_percent": {"warning": 85, "critical": 95},
            "response_time": {"warning": 2.0, "critical": 5.0},
            "error_rate": {"warning": 0.05, "critical": 0.1}
        }
    
    def start_monitoring(self):
        """Start health monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._collect_system_metrics()
                self._check_component_health()
                self._evaluate_alerts()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self._update_metric("cpu_percent", cpu_percent, "%", "CPU usage percentage")
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self._update_metric("memory_percent", memory.percent, "%", "Memory usage percentage")
            self._update_metric("memory_available", memory.available / (1024**3), "GB", "Available memory")
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._update_metric("disk_percent", disk_percent, "%", "Disk usage percentage")
            
            # Network metrics
            net_io = psutil.net_io_counters()
            self._update_metric("network_bytes_sent", net_io.bytes_sent, "bytes", "Network bytes sent")
            self._update_metric("network_bytes_recv", net_io.bytes_recv, "bytes", "Network bytes received")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _update_metric(self, name: str, value: float, unit: str, description: str):
        """Update a health metric"""
        thresholds = self.thresholds.get(name, {"warning": float('inf'), "critical": float('inf')})
        
        # Determine status
        if value >= thresholds["critical"]:
            status = HealthStatus.CRITICAL
        elif value >= thresholds["warning"]:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        metric = HealthMetric(
            name=name,
            value=value,
            status=status,
            threshold_warning=thresholds["warning"],
            threshold_critical=thresholds["critical"],
            timestamp=datetime.now(),
            unit=unit,
            description=description
        )
        
        self.metrics[name] = metric
    
    def _check_component_health(self):
        """Check health of individual components"""
        components = {
            "speech": self._check_speech_health,
            "vision": self._check_vision_health,
            "nlp": self._check_nlp_health,
            "system": self._check_system_health
        }
        
        for component_name, check_func in components.items():
            try:
                health_status = check_func()
                self._update_metric(f"{component_name}_health", 
                                  1.0 if health_status else 0.0, 
                                  "", f"{component_name} component health")
            except Exception as e:
                logger.error(f"Error checking {component_name} health: {e}")
                self._update_metric(f"{component_name}_health", 0.0, "", f"{component_name} component health")
    
    def _check_speech_health(self) -> bool:
        """Check speech component health"""
        try:
            from modules.speech import speech_processor
            return speech_processor.initialized if hasattr(speech_processor, 'initialized') else True
        except Exception:
            return False
    
    def _check_vision_health(self) -> bool:
        """Check vision component health"""
        try:
            from modules.vision import vision_system
            return vision_system.is_processing if hasattr(vision_system, 'is_processing') else True
        except Exception:
            return False
    
    def _check_nlp_health(self) -> bool:
        """Check NLP component health"""
        try:
            from modules.nlp import nlp_processor
            return nlp_processor.initialized if hasattr(nlp_processor, 'initialized') else True
        except Exception:
            return False
    
    def _check_system_health(self) -> bool:
        """Check overall system health"""
        try:
            # Check if critical metrics are within acceptable ranges
            cpu_metric = self.metrics.get("cpu_percent")
            memory_metric = self.metrics.get("memory_percent")
            
            if cpu_metric and cpu_metric.status == HealthStatus.CRITICAL:
                return False
            if memory_metric and memory_metric.status == HealthStatus.CRITICAL:
                return False
            
            return True
        except Exception:
            return False
    
    def _evaluate_alerts(self):
        """Evaluate and generate alerts based on metrics"""
        for metric_name, metric in self.metrics.items():
            if metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                alert_id = f"{metric_name}_{metric.status.value}_{int(metric.timestamp.timestamp())}"
                
                # Check if alert already exists
                existing_alert = next((a for a in self.alerts if a.id == alert_id), None)
                if not existing_alert:
                    alert = SystemAlert(
                        id=alert_id,
                        severity=metric.status.value,
                        message=f"{metric.description}: {metric.value}{metric.unit} (threshold: {metric.threshold_warning if metric.status == HealthStatus.WARNING else metric.threshold_critical}{metric.unit})",
                        component=metric_name,
                        timestamp=metric.timestamp
                    )
                    
                    self.alerts.append(alert)
                    self._notify_alert(alert)
    
    def _notify_alert(self, alert: SystemAlert):
        """Notify registered callbacks about new alert"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def register_alert_callback(self, callback: Callable[[SystemAlert], None]):
        """Register callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        current_time = datetime.now()
        
        # Count metrics by status
        status_counts = {status.value: 0 for status in HealthStatus}
        for metric in self.metrics.values():
            status_counts[metric.status.value] += 1
        
        # Get recent alerts
        recent_alerts = [
            asdict(alert) for alert in self.alerts 
            if current_time - alert.timestamp < timedelta(hours=24)
        ]
        
        # Overall health score (0-100)
        total_metrics = len(self.metrics)
        if total_metrics > 0:
            health_score = (
                (status_counts["healthy"] * 100 + 
                 status_counts["warning"] * 50 + 
                 status_counts["critical"] * 0) / total_metrics
            )
        else:
            health_score = 0
        
        return {
            "overall_health_score": health_score,
            "status_distribution": status_counts,
            "total_metrics": total_metrics,
            "recent_alerts": recent_alerts,
            "last_check": current_time.isoformat(),
            "monitoring_active": self.is_monitoring
        }
    
    def get_metric(self, name: str) -> Optional[HealthMetric]:
        """Get specific metric"""
        return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, HealthMetric]:
        """Get all current metrics"""
        return self.metrics.copy()
    
    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved"""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                logger.info(f"Alert {alert_id} resolved")
                break

# Global health monitor
health_monitor = HealthMonitor()