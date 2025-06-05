"""
Enhanced neural network brain module for Jarvis AI Assistant.

Features:
- Grok AI-powered reasoning and problem-solving
- Transformer-based pattern recognition
- Priority-based memory management
- Advanced context analysis
- Cross-module integration
"""

import os
import json
import threading
import heapq
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
import logging
from queue import PriorityQueue
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from modules.grok_client import GrokClient
from modules.vision import VisionProcessor
from modules.speech import SpeechProcessor
from modules.nlp import NLPProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class MemoryItem:
    key: str
    value: Any
    priority: MemoryPriority
    timestamp: datetime
    category: str
    access_count: int
    last_access: datetime
    metadata: Dict

    def __lt__(self, other):
        return self.priority.value < other.priority.value

class TransformerBrain(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_heads: int):
        super(TransformerBrain, self).__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        
        # Output layers
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, input_size)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # Input embedding
        x = self.input_embedding(x)
        x = self.norm(x)
        
        # Transformer processing
        x = self.transformer(x)
        
        # Output processing
        x = self.fc(x)
        x = torch.relu(x)
        x = self.output(x)
        
        return x

class ContextAnalyzer:
    def __init__(self):
        self.nlp = NLPProcessor()
    
    def analyze_context(self, 
                       current_request: str,
                       conversation_history: List[Dict],
                       memory_items: List[MemoryItem]) -> Dict:
        """Perform deep context analysis"""
        
        # Extract key information from current request
        request_analysis = self.nlp.analyze_text(current_request)
        
        # Analyze conversation flow
        conversation_context = self._analyze_conversation(conversation_history)
        
        # Find relevant memories
        relevant_memories = self._find_relevant_memories(
            request_analysis,
            memory_items
        )
        
        return {
            "request_analysis": request_analysis,
            "conversation_context": conversation_context,
            "relevant_memories": relevant_memories,
            "predicted_intent": self._predict_intent(
                request_analysis,
                conversation_context
            )
        }
    
    def _analyze_conversation(self, history: List[Dict]) -> Dict:
        """Analyze conversation flow and patterns"""
        if not history:
            return {"flow": "new", "context_depth": 0}
            
        topics = []
        sentiment_flow = []
        
        for entry in history[-5:]:  # Analyze last 5 interactions
            analysis = self.nlp.analyze_text(entry["request"])
            topics.extend(analysis["topics"])
            sentiment_flow.append(analysis["sentiment"])
        
        return {
            "flow": "ongoing",
            "context_depth": len(history),
            "main_topics": list(set(topics)),
            "sentiment_trend": np.mean(sentiment_flow)
        }
    
    def _find_relevant_memories(self, 
                              request_analysis: Dict,
                              memories: List[MemoryItem]) -> List[MemoryItem]:
        """Find memories relevant to current context"""
        relevant = []
        for memory in memories:
            relevance_score = self._calculate_relevance(
                request_analysis,
                memory
            )
            if relevance_score > 0.5:  # Threshold for relevance
                relevant.append(memory)
        return sorted(relevant, key=lambda x: x.priority.value)
    
    def _predict_intent(self, 
                       request_analysis: Dict,
                       conversation_context: Dict) -> str:
        """Predict user intent based on context"""
        # Combine current request with conversation context
        return self.nlp.predict_intent(
            request_analysis["text"],
            context=conversation_context
        )
    
    def _calculate_relevance(self, 
                           analysis: Dict,
                           memory: MemoryItem) -> float:
        """Calculate relevance score between current context and memory"""
        # Compare topics
        topic_overlap = len(
            set(analysis["topics"]) & 
            set(self.nlp.extract_topics(str(memory.value)))
        )
        
        # Consider recency
        time_factor = 1.0 / (1.0 + (
            datetime.now() - memory.last_access
        ).total_seconds() / 86400)  # Days factor
        
        # Consider access frequency
        frequency_factor = min(1.0, memory.access_count / 10.0)
        
        return (0.4 * topic_overlap + 
                0.3 * time_factor + 
                0.3 * frequency_factor)

class BrainManager:
    def __init__(self, 
                 storage_path: str,
                 input_size: int = 256,
                 hidden_size: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

        # Initialize neural network
        self.model = TransformerBrain(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        # Initialize memory system
        self.memory_queue = PriorityQueue()
        self.memory_dict = {}
        self.memory_lock = threading.Lock()

        # Initialize context analyzer
        self.context_analyzer = ContextAnalyzer()

        # Initialize module interfaces
        self.vision = VisionProcessor()
        self.speech = SpeechProcessor()
        self.nlp = NLPProcessor()
        self.grok = GrokClient()

        # Load existing memory
        self._load_memory()

        # Conversation tracking
        self.conversation_history = []
        self.max_history = 10

    def _load_memory(self):
        """Load memory from storage"""
        memory_file = os.path.join(self.storage_path, "memory.json")
        if os.path.exists(memory_file):
            with open(memory_file, "r") as f:
                data = json.load(f)
                for key, item in data.items():
                    memory_item = MemoryItem(
                        key=key,
                        value=item["value"],
                        priority=MemoryPriority[item["priority"]],
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        category=item["category"],
                        access_count=item["access_count"],
                        last_access=datetime.fromisoformat(item["last_access"]),
                        metadata=item["metadata"]
                    )
                    self.memory_dict[key] = memory_item
                    self.memory_queue.put(memory_item)

    def _save_memory(self):
        """Save memory to storage"""
        memory_file = os.path.join(self.storage_path, "memory.json")
        data = {}
        for key, item in self.memory_dict.items():
            data[key] = {
                "value": item.value,
                "priority": item.priority.name,
                "timestamp": item.timestamp.isoformat(),
                "category": item.category,
                "access_count": item.access_count,
                "last_access": item.last_access.isoformat(),
                "metadata": item.metadata
            }
        with open(memory_file, "w") as f:
            json.dump(data, f, indent=2)

    def remember(self, 
                key: str,
                value: Any,
                priority: MemoryPriority = MemoryPriority.MEDIUM,
                category: str = "general",
                metadata: Dict = None):
        """Store information in memory with priority"""
        with self.memory_lock:
            memory_item = MemoryItem(
                key=key,
                value=value,
                priority=priority,
                timestamp=datetime.now(),
                category=category,
                access_count=0,
                last_access=datetime.now(),
                metadata=metadata or {}
            )
            self.memory_dict[key] = memory_item
            self.memory_queue.put(memory_item)
            self._save_memory()

    def recall(self, key: str) -> Optional[Any]:
        """Retrieve information from memory"""
        with self.memory_lock:
            if key in self.memory_dict:
                item = self.memory_dict[key]
                item.access_count += 1
                item.last_access = datetime.now()
                self._save_memory()
                return item.value
            return None

    def search_memory(self, query: str) -> List[Tuple[str, Any]]:
        """Search memory for items matching the query"""
        results = []
        with self.memory_lock:
            for key, item in self.memory_dict.items():
                # Search in key
                if query.lower() in key.lower():
                    results.append((key, item.value))
                    continue
                
                # Search in value
                if isinstance(item.value, (str, dict)):
                    value_str = str(item.value)
                    if query.lower() in value_str.lower():
                        results.append((key, item.value))
                        continue
                
                # Search in metadata if available
                if item.metadata:
                    metadata_str = str(item.metadata)
                    if query.lower() in metadata_str.lower():
                        results.append((key, item.value))
                        continue
        
        return results

    def process_request(self, request: str, context: Dict = None) -> str:
        """Process a request using all available capabilities"""
        
        # Analyze context
        context_analysis = self.context_analyzer.analyze_context(
            request,
            self.conversation_history,
            list(self.memory_dict.values())
        )
        
        # Prepare search parameters
        search_params = {
            "mode": "auto",
            "return_citations": True,
            "max_search_results": 5
        }

        # Build context message
        context_message = self._build_context(context_analysis)
        
        # Prepare messages for Grok
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Jarvis, an advanced AI assistant with access to various capabilities "
                    "including system control, web browsing, news fetching, and learning from interactions. "
                    f"\nContext: {context_message}"
                )
            },
            {"role": "user", "content": request}
        ]

        # Get response from Grok
        response = self.grok.chat(
            messages=messages,
            search_parameters=search_params,
            temperature=0.7
        )

        # Update conversation history
        self._update_conversation_history(request, response, context_analysis)

        # Learn from interaction
        self.learn_from_interaction(request, response, context_analysis)

        return response

    def _build_context(self, context_analysis: Dict) -> str:
        """Build context string using analysis results"""
        context_parts = []
        
        # Add conversation flow information
        if context_analysis["conversation_context"]["flow"] == "ongoing":
            context_parts.append("Ongoing conversation about: " + 
                               ", ".join(context_analysis["conversation_context"]["main_topics"]))
        
        # Add relevant memories
        if context_analysis["relevant_memories"]:
            context_parts.append("\nRelevant knowledge:")
            for memory in context_analysis["relevant_memories"][:3]:  # Top 3 most relevant
                context_parts.append(f"- {memory.key}: {memory.value}")
        
        # Add predicted intent
        context_parts.append(f"\nPredicted intent: {context_analysis['predicted_intent']}")
        
        return "\n".join(context_parts)

    def _update_conversation_history(self, 
                                  request: str,
                                  response: str,
                                  context_analysis: Dict):
        """Update conversation history with new interaction"""
        self.conversation_history.append({
            "request": request,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "context_analysis": context_analysis
        })
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def learn_from_interaction(self, 
                             request: str,
                             response: str,
                             context_analysis: Dict):
        """Enhanced learning from interactions"""
        try:
            # Extract patterns for neural network training
            input_features = self.nlp.extract_features(request)
            target_features = self.nlp.extract_features(response)
            
            # Convert to tensors
            input_tensor = torch.FloatTensor(input_features).unsqueeze(0)
            target_tensor = torch.FloatTensor(target_features).unsqueeze(0)
            
            # Train neural network
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(input_tensor)
            loss = self.criterion(output, target_tensor)
            loss.backward()
            self.optimizer.step()
            
            # Store interaction in memory
            interaction_key = f"interaction_{datetime.now().isoformat()}"
            self.remember(
                key=interaction_key,
                value={
                    "request": request,
                    "response": response,
                    "context_analysis": context_analysis,
                    "learning_loss": loss.item()
                },
                priority=MemoryPriority.MEDIUM,
                category="interactions",
                metadata={
                    "topics": context_analysis["request_analysis"]["topics"],
                    "sentiment": context_analysis["request_analysis"]["sentiment"]
                }
            )

        except Exception as e:
            logger.error(f"Error learning from interaction: {e}")

    def process_multimodal_input(self, 
                               text: str = None,
                               image: np.ndarray = None,
                               audio: np.ndarray = None) -> str:
        """Process input from multiple modalities"""
        context = {}
        
        try:
            # Process visual input if available
            if image is not None and isinstance(image, np.ndarray):
                visual_analysis = self.vision.analyze_image(image)
                # Convert numpy arrays to lists for JSON serialization
                if "embedding" in visual_analysis:
                    visual_analysis["embedding"] = visual_analysis["embedding"].tolist()
                context["visual_context"] = visual_analysis
            
            # Process audio input if available
            if audio is not None and isinstance(audio, np.ndarray):
                audio_analysis = self.speech.analyze_audio(audio)
                # Convert numpy arrays to lists for JSON serialization
                if "features" in audio_analysis:
                    audio_analysis["features"] = audio_analysis["features"].tolist()
                context["audio_context"] = audio_analysis
            
            # Process text with additional context
            return self.process_request(text or "", context)
            
        except Exception as e:
            logger.error(f"Error in multimodal processing: {e}")
            return "Error processing multimodal input"

    def get_capabilities(self) -> Dict[str, Any]:
        """Return current capabilities and learning status"""
        return {
            "neural_network": {
                "architecture": "TransformerBrain",
                "input_size": self.model.input_embedding.in_features,
                "hidden_size": self.model.fc.in_features,
                "num_layers": len(self.model.transformer.layers),
                "num_heads": self.model.transformer.layers[0].self_attn.num_heads
            },
            "memory": {
                "size": len(self.memory_dict),
                "categories": list(set(m.category for m in self.memory_dict.values())),
                "priority_distribution": {
                    priority.name: len([m for m in self.memory_dict.values() 
                                     if m.priority == priority])
                    for priority in MemoryPriority
                }
            },
            "conversation": {
                "history_length": len(self.conversation_history),
                "max_history": self.max_history
            },
            "integrations": {
                "grok_available": self.grok.test_connection(),
                "vision_available": self.vision is not None,
                "speech_available": self.speech is not None,
                "nlp_available": self.nlp is not None
            }
        }

# Initialize brain manager with storage path
brain_storage_path = "E:/JarvisBrainStorage"
brain_manager = BrainManager(storage_path=brain_storage_path)
