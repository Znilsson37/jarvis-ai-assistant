"""Centralized integration manager for coordinating all Jarvis components"""

import asyncio
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ComponentState(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class ComponentStatus:
    name: str
    state: ComponentState
    last_activity: float
    error_count: int
    performance_metrics: Dict[str, Any]

class IntegrationManager:
    """Centralized manager for coordinating all Jarvis components"""
    
    def __init__(self):
        self.components = {}
        self.event_bus = {}
        self.is_running = False
        self.coordination_thread = None
        
        # Component dependencies
        self.dependencies = {
            "speech": [],
            "vision": [],
            "nlp": ["speech"],
            "system_control": [],
            "browser": [],
            "ui": ["speech", "vision"]
        }
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components with proper dependency order"""
        # Sort components by dependency order
        sorted_components = self._topological_sort(self.dependencies)
        
        for component_name in sorted_components:
            try:
                self._initialize_component(component_name)
            except Exception as e:
                logger.error(f"Failed to initialize {component_name}: {e}")
                self.components[component_name] = ComponentStatus(
                    name=component_name,
                    state=ComponentState.ERROR,
                    last_activity=time.time(),
                    error_count=1,
                    performance_metrics={}
                )
    
    def _topological_sort(self, dependencies: Dict[str, List[str]]) -> List[str]:
        """Sort components by dependency order"""
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected involving {node}")
            if node in visited:
                return
            
            temp_visited.add(node)
            for dependency in dependencies.get(node, []):
                visit(dependency)
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)
        
        for component in dependencies:
            if component not in visited:
                visit(component)
        
        return result
    
    def _initialize_component(self, component_name: str):
        """Initialize a specific component"""
        logger.info(f"Initializing component: {component_name}")
        
        self.components[component_name] = ComponentStatus(
            name=component_name,
            state=ComponentState.INITIALIZING,
            last_activity=time.time(),
            error_count=0,
            performance_metrics={}
        )
        
        try:
            if component_name == "speech":
                from modules.enhanced_speech import enhanced_speech
                enhanced_speech.register_callback("transcription_complete", 
                                                self._on_speech_transcription)
                enhanced_speech.register_callback("speech_complete", 
                                                self._on_speech_synthesis)
                
            elif component_name == "vision":
                from modules.vision import vision_system
                # Initialize vision system
                
            elif component_name == "nlp":
                from modules.nlp import nlp_processor
                # Initialize NLP processor
                
            elif component_name == "system_control":
                from modules.secure_system_control import secure_system
                # Initialize system control
                
            elif component_name == "browser":
                # Initialize browser control
                pass
                
            elif component_name == "ui":
                from modules.ui_visualization import initialize
                initialize()
            
            self.components[component_name].state = ComponentState.READY
            logger.info(f"Component {component_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing {component_name}: {e}")
            self.components[component_name].state = ComponentState.ERROR
            self.components[component_name].error_count += 1
            raise
    
    def start_coordination(self):
        """Start the coordination system"""
        if self.is_running:
            return
        
        self.is_running = True
        self.coordination_thread = threading.Thread(target=self._coordination_loop)
        self.coordination_thread.daemon = True
        self.coordination_thread.start()
        
        logger.info("Integration coordination started")
    
    def stop_coordination(self):
        """Stop the coordination system"""
        self.is_running = False
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5)
        
        logger.info("Integration coordination stopped")
    
    def _coordination_loop(self):
        """Main coordination loop"""
        while self.is_running:
            try:
                self._check_component_health()
                self._handle_component_communication()
                self._update_performance_metrics()
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                time.sleep(5)  # Longer delay on error
    
    def _check_component_health(self):
        """Check health of all components"""
        current_time = time.time()
        
        for component_name, status in self.components.items():
            # Check if component is responsive
            if current_time - status.last_activity > 60:  # 1 minute timeout
                if status.state == ComponentState.READY:
                    logger.warning(f"Component {component_name} appears unresponsive")
                    status.state = ComponentState.ERROR
                    status.error_count += 1
    
    def _handle_component_communication(self):
        """Handle inter-component communication"""
        # Process events in the event bus
        for event_type, handlers in self.event_bus.items():
            # Implementation for event processing
            pass
    
    def _update_performance_metrics(self):
        """Update performance metrics for all components"""
        for component_name, status in self.components.items():
            try:
                metrics = self._get_component_metrics(component_name)
                status.performance_metrics.update(metrics)
            except Exception as e:
                logger.error(f"Error updating metrics for {component_name}: {e}")
    
    def _get_component_metrics(self, component_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific component"""
        metrics = {}
        
        try:
            if component_name == "speech":
                from modules.enhanced_speech import enhanced_speech
                metrics = enhanced_speech.get_metrics()
                
            elif component_name == "vision":
                # Get vision metrics
                pass
                
            elif component_name == "nlp":
                # Get NLP metrics
                pass
                
        except Exception as e:
            logger.error(f"Error getting metrics for {component_name}: {e}")
        
        return metrics
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_bus:
            self.event_bus[event_type] = []
        self.event_bus[event_type].append(handler)
    
    def emit_event(self, event_type: str, data: Any):
        """Emit event to all registered handlers"""
        if event_type in self.event_bus:
            for handler in self.event_bus[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
    
    def _on_speech_transcription(self, result):
        """Handle speech transcription completion"""
        self.components["speech"].last_activity = time.time()
        self.emit_event("speech_transcribed", result)
    
    def _on_speech_synthesis(self, audio_path):
        """Handle speech synthesis completion"""
        self.components["speech"].last_activity = time.time()
        self.emit_event("speech_synthesized", audio_path)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "coordination_active": self.is_running,
            "components": {},
            "overall_health": "healthy"
        }
        
        error_count = 0
        total_components = len(self.components)
        
        for name, component in self.components.items():
            status["components"][name] = {
                "state": component.state.value,
                "last_activity": component.last_activity,
                "error_count": component.error_count,
                "performance_metrics": component.performance_metrics
            }
            
            if component.state == ComponentState.ERROR:
                error_count += 1
        
        # Determine overall health
        if error_count == 0:
            status["overall_health"] = "healthy"
        elif error_count < total_components / 2:
            status["overall_health"] = "degraded"
        else:
            status["overall_health"] = "critical"
        
        return status
    
    def restart_component(self, component_name: str) -> bool:
        """Restart a specific component"""
        try:
            logger.info(f"Restarting component: {component_name}")
            
            # Shutdown component
            self._shutdown_component(component_name)
            
            # Reinitialize component
            self._initialize_component(component_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart {component_name}: {e}")
            return False
    
    def _shutdown_component(self, component_name: str):
        """Shutdown a specific component"""
        if component_name in self.components:
            self.components[component_name].state = ComponentState.SHUTDOWN
            
            # Component-specific shutdown logic
            try:
                if component_name == "speech":
                    from modules.enhanced_speech import enhanced_speech
                    enhanced_speech.stop_continuous_listening()
                    
                elif component_name == "vision":
                    from modules.vision import vision_system
                    vision_system.stop_vision_processing()
                    
                # Add other component shutdown logic
                
            except Exception as e:
                logger.error(f"Error shutting down {component_name}: {e}")

# Global integration manager
integration_manager = IntegrationManager()