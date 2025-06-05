"""Integration tests for enhanced NLP module"""

import unittest
import numpy as np
from modules.nlp import NLPProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestNLPIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize NLP processor for all tests"""
        cls.nlp = NLPProcessor()
        
    def test_initialization(self):
        """Test NLP processor initialization"""
        self.assertTrue(hasattr(self.nlp, 'initialized'))
        if self.nlp.initialized:
            self.assertIsNotNone(self.nlp.tokenizer)
            self.assertIsNotNone(self.nlp.model)
            self.assertIsNotNone(self.nlp.sentiment_analyzer)
    
    def test_embedding_generation(self):
        """Test text embedding generation"""
        test_text = "Hello, this is a test sentence."
        embedding = self.nlp.get_embedding(test_text)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape[0], self.nlp.embedding_size)
        self.assertTrue(np.any(embedding))  # Check if embedding contains non-zero values
    
    def test_intent_recognition(self):
        """Test intent recognition for various commands"""
        test_cases = [
            {
                "text": "open chrome browser",
                "expected_intent": "launch_app"
            },
            {
                "text": "what's the system status",
                "expected_intent": "system_control"
            },
            {
                "text": "search for latest news",
                "expected_intent": "web_search"
            },
            {
                "text": "check my email",
                "expected_intent": "email"
            }
        ]
        
        for test in test_cases:
            result = self.nlp.recognize_intent(test["text"])
            self.assertIsInstance(result, dict)
            self.assertIn("intent", result)
            self.assertIn("confidence", result)
            self.assertEqual(result["intent"], test["expected_intent"],
                           f"Failed to recognize intent for: {test['text']}")
            self.assertGreater(result["confidence"], 0.0)
    
    def test_entity_extraction(self):
        """Test entity extraction for various text inputs"""
        test_cases = [
            {
                "text": "send email to john@example.com",
                "expected_entities": {"email": ["john@example.com"]}
            },
            {
                "text": "open chrome browser",
                "expected_entities": {"browser_app": ["chrome"]}
            },
            {
                "text": "set volume to 50",
                "expected_entities": {"number": ["50"]}
            }
        ]
        
        for test in test_cases:
            entities = self.nlp.extract_entities(test["text"])
            self.assertIsInstance(entities, dict)
            for key, values in test["expected_entities"].items():
                self.assertIn(key, entities)
                for value in values:
                    self.assertIn(value, entities[key])
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis"""
        test_cases = [
            {
                "text": "This is great and wonderful!",
                "expected_sentiment": "positive"
            },
            {
                "text": "This is terrible and horrible.",
                "expected_sentiment": "negative"
            },
            {
                "text": "The weather is cloudy today.",
                "expected_sentiment": "neutral"
            }
        ]
        
        for test in test_cases:
            sentiment = self.nlp.analyze_sentiment(test["text"])
            self.assertIsInstance(sentiment, float)
            self.assertTrue(-1.0 <= sentiment <= 1.0)
            
            if test["expected_sentiment"] == "positive":
                self.assertGreater(sentiment, 0)
            elif test["expected_sentiment"] == "negative":
                self.assertLess(sentiment, 0)
            else:
                self.assertAlmostEqual(abs(sentiment), 0, delta=0.3)
    
    def test_context_management(self):
        """Test conversation context management"""
        # Test context window updates
        test_conversation = [
            "Hello, how are you?",
            "What's the weather like?",
            "Can you open Chrome?",
            "Search for news about AI"
        ]
        
        for text in test_conversation:
            self.nlp._update_context(text)
        
        self.assertLessEqual(len(self.nlp.context_window), self.nlp.max_context_items)
        
        # Test conversation flow analysis
        flow_analysis = self.nlp._analyze_conversation_flow()
        self.assertIsInstance(flow_analysis, dict)
        self.assertIn("flow_type", flow_analysis)
        self.assertIn("topic_consistency", flow_analysis)
        
        # Test context relevance
        relevance = self.nlp._calculate_context_relevance("What's the latest news?")
        self.assertIsInstance(relevance, float)
        self.assertTrue(0 <= relevance <= 1)
    
    def test_task_planning(self):
        """Test task planning functionality"""
        test_cases = [
            {
                "command": "open chrome and search for weather",
                "expected_steps": ["launch_browser", "search"]
            },
            {
                "command": "check system status",
                "expected_steps": ["check_system"]
            }
        ]
        
        for test in test_cases:
            result = self.nlp.plan_task(test["command"])
            
            self.assertIsInstance(result, dict)
            self.assertIn("steps", result)
            self.assertIn("estimated_time", result)
            
            steps = result["steps"]
            self.assertIsInstance(steps, list)
            self.assertTrue(steps)  # Check if steps is not empty
            
            # Check if expected steps are present
            step_actions = [step["action"] for step in steps]
            for expected_step in test["expected_steps"]:
                self.assertTrue(
                    any(expected_step in action for action in step_actions),
                    f"Expected step {expected_step} not found in {step_actions}"
                )
    
    def test_text_similarity(self):
        """Test semantic similarity calculation"""
        text1 = "What's the weather like today?"
        text2 = "How's the weather today?"
        text3 = "Open Chrome browser."
        
        # Similar texts should have high similarity
        similarity_similar = self.nlp.calculate_similarity(text1, text2)
        self.assertGreater(similarity_similar, 0.7)
        
        # Different texts should have low similarity
        similarity_different = self.nlp.calculate_similarity(text1, text3)
        self.assertLess(similarity_different, 0.5)
    
    def test_comprehensive_analysis(self):
        """Test comprehensive text analysis"""
        test_text = "Can you open Chrome and search for the weather forecast?"
        
        analysis = self.nlp.analyze_text(test_text)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn("text", analysis)
        self.assertIn("topics", analysis)
        self.assertIn("sentiment", analysis)
        self.assertIn("entities", analysis)
        self.assertIn("intent", analysis)
        self.assertIn("embedding", analysis)
        
        # Check if topics are extracted
        self.assertIsInstance(analysis["topics"], list)
        self.assertTrue(analysis["topics"])
        
        # Check if entities are extracted
        self.assertIsInstance(analysis["entities"], dict)
        self.assertIn("browser_app", analysis["entities"])
        
        # Check if intent is recognized
        self.assertEqual(analysis["intent"]["intent"], "launch_app")
        
        # Check if embedding is present
        self.assertIsInstance(analysis["embedding"], list)
        self.assertEqual(len(analysis["embedding"]), self.nlp.embedding_size)

if __name__ == '__main__':
    unittest.main(verbosity=2)
