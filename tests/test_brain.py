import unittest
from unittest.mock import patch, MagicMock
import json
import os
import shutil
import tempfile
import torch
import numpy as np
from datetime import datetime

from modules.brain import BrainManager, MemoryPriority, MemoryItem, ContextAnalyzer

class TestBrainManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.brain = BrainManager(storage_path=self.test_dir)

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    def test_transformer_network(self):
        """Test transformer neural network operations"""
        # Test input shape
        input_tensor = torch.randn(1, 256)  # Batch size 1, input size 256
        output = self.brain.model(input_tensor)
        self.assertEqual(output.shape, (1, 256))
        
        # Test training
        target_tensor = torch.randn(1, 256)
        self.brain.model.train()
        self.brain.optimizer.zero_grad()
        output = self.brain.model(input_tensor)
        loss = self.brain.criterion(output, target_tensor)
        loss.backward()
        self.assertIsInstance(loss.item(), float)

    def test_priority_memory(self):
        """Test priority-based memory operations"""
        # Test storing with different priorities
        self.brain.remember(
            "critical_info",
            "Important data",
            priority=MemoryPriority.CRITICAL
        )
        self.brain.remember(
            "low_priority",
            "Less important",
            priority=MemoryPriority.LOW
        )
        
        # Verify memory storage
        critical_value = self.brain.recall("critical_info")
        low_value = self.brain.recall("low_priority")
        self.assertEqual(critical_value, "Important data")
        self.assertEqual(low_value, "Less important")
        
        # Verify priority queue ordering
        memory_items = []
        while not self.brain.memory_queue.empty():
            memory_items.append(self.brain.memory_queue.get())
        
        self.assertEqual(len(memory_items), 2)
        self.assertEqual(memory_items[0].priority, MemoryPriority.CRITICAL)

    def test_context_analysis(self):
        """Test context analysis capabilities"""
        # Setup test conversation history
        self.brain.conversation_history = [
            {
                "request": "What's the weather like?",
                "response": "It's sunny today",
                "timestamp": datetime.now().isoformat()
            },
            {
                "request": "Will it rain tomorrow?",
                "response": "Yes, rain is expected",
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        # Add test memories
        self.brain.remember(
            "weather_pref",
            "User prefers sunny weather",
            category="preferences"
        )
        
        # Test context analysis
        context_analysis = self.brain.context_analyzer.analyze_context(
            "Should I bring an umbrella?",
            self.brain.conversation_history,
            list(self.brain.memory_dict.values())
        )
        
        # Verify analysis results
        self.assertIn("request_analysis", context_analysis)
        self.assertIn("conversation_context", context_analysis)
        self.assertIn("relevant_memories", context_analysis)
        self.assertIn("predicted_intent", context_analysis)

    @patch('modules.vision.VisionProcessor')
    @patch('modules.speech.SpeechProcessor')
    def test_multimodal_processing(self, mock_speech_class, mock_vision_class):
        """Test processing of multiple input modalities"""
        # Create mock instances
        mock_vision = MagicMock()
        mock_speech = MagicMock()
        
        # Configure mock returns
        mock_vision.analyze_image.return_value = {
            "objects": ["umbrella", "rain"],
            "scene": "outdoor"
        }
        mock_speech.analyze_audio.return_value = {
            "transcript": "Will I need an umbrella?",
            "confidence": 0.95
        }
        
        # Set up mock classes to return our configured mocks
        mock_vision_class.return_value = mock_vision
        mock_speech_class.return_value = mock_speech
        
        # Replace the brain's processors with our mocks
        self.brain.vision = mock_vision
        self.brain.speech = mock_speech
        
        # Test with multiple inputs
        image_data = np.random.rand(100, 100, 3)  # Mock image
        audio_data = np.random.rand(1000)  # Mock audio
        
        response = self.brain.process_multimodal_input(
            text="Should I bring an umbrella?",
            image=image_data,
            audio=audio_data
        )
        
        # Verify multimodal processing
        mock_vision.analyze_image.assert_called_once()
        mock_speech.analyze_audio.assert_called_once()

    def test_learning_from_interaction(self):
        """Test enhanced learning capabilities"""
        request = "What's the weather forecast?"
        response = "It will be sunny with a high of 75°F"
        
        # Create mock context analysis
        context_analysis = {
            "request_analysis": {
                "text": "What's the weather forecast?",
                "topics": ["weather", "forecast"],
                "sentiment": 0.5
            },
            "conversation_context": {
                "flow": "new",
                "context_depth": 1
            },
            "relevant_memories": []
        }
        
        # Test learning
        self.brain.learn_from_interaction(request, response, context_analysis)
        
        # Verify learning results
        results = self.brain.search_memory("weather")
        self.assertGreater(len(results), 0)
        
        # Verify stored interaction
        found = False
        for key, value in results:
            if isinstance(value, dict) and value.get("request") == request:
                found = True
                self.assertEqual(value["response"], response)
                self.assertIn("learning_loss", value)
                break
        
        self.assertTrue(found, "Interaction was not properly stored")

    def test_capabilities_report(self):
        """Test capability reporting"""
        capabilities = self.brain.get_capabilities()
        
        # Verify neural network configuration
        self.assertIn("neural_network", capabilities)
        self.assertEqual(capabilities["neural_network"]["architecture"], "TransformerBrain")
        self.assertEqual(capabilities["neural_network"]["input_size"], 256)
        
        # Verify memory system status
        self.assertIn("memory", capabilities)
        self.assertIn("priority_distribution", capabilities["memory"])
        
        # Verify integration status
        self.assertIn("integrations", capabilities)
        self.assertIn("grok_available", capabilities["integrations"])
        self.assertIn("vision_available", capabilities["integrations"])
        self.assertIn("speech_available", capabilities["integrations"])

    @patch('modules.grok_client.GrokClient')
    def test_process_request(self, MockGrokClient):
        """Test request processing with context"""
        # Create mock instance
        mock_grok = MagicMock()
        mock_grok.chat.return_value = "Test response with context"
        MockGrokClient.return_value = mock_grok
        
        # Replace the brain's Grok client
        self.brain.grok = mock_grok
        
        # Add some context
        self.brain.remember(
            "user_preference",
            "Prefers detailed explanations",
            priority=MemoryPriority.HIGH
        )
        
        # Process request
        response = self.brain.process_request("Tell me about neural networks")
        
        # Verify Grok interaction
        mock_grok.chat.assert_called_once()
        call_args = mock_grok.chat.call_args[1]
        
        # Verify context inclusion
        messages = call_args["messages"]
        system_message = messages[0]["content"]
        self.assertIn("Context:", system_message)
        
        # Verify search parameters
        self.assertIn("search_parameters", call_args)
        self.assertEqual(call_args["search_parameters"]["mode"], "auto")

    def test_memory_persistence(self):
        """Test memory persistence across sessions"""
        # Store test memories
        self.brain.remember(
            "test_persistence",
            "Persistent data",
            priority=MemoryPriority.HIGH,
            category="test"
        )
        
        # Create new brain instance with same storage
        new_brain = BrainManager(storage_path=self.test_dir)
        
        # Verify memory persistence
        value = new_brain.recall("test_persistence")
        self.assertEqual(value, "Persistent data")
        
        # Verify memory metadata
        memory_item = new_brain.memory_dict["test_persistence"]
        self.assertEqual(memory_item.priority, MemoryPriority.HIGH)
        self.assertEqual(memory_item.category, "test")

if __name__ == '__main__':
    unittest.main()
