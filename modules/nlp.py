"""Natural Language Processing module for Jarvis with enhanced capabilities"""

from typing import Dict, List, Any, Optional
import numpy as np
import logging
from datetime import datetime
import re
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    pipeline,
    AutoModelForSequenceClassification
)
import torch
from modules.grok_client import GrokClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPProcessor:
    def __init__(self):
        self.initialized = False
        self.embedding_size = 768  # BERT base embedding size
        self.context_window = []
        self.max_context_items = 5
        self.grok_client = GrokClient()
        
        # Initialize models
        try:
            logger.info("Initializing NLP models...")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased")
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            self.ner_pipeline = pipeline("ner")
            self.initialized = True
            logger.info("NLP models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
            # Fall back to basic processing if model initialization fails
            self.initialized = False
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive NLP analysis on text"""
        try:
            # Update context window
            self._update_context(text)
            
            analysis = {
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "topics": self.extract_topics(text),
                "sentiment": self.analyze_sentiment(text),
                "entities": self.extract_entities(text),
                "intent": self.recognize_intent(text),
                "embedding": self.get_embedding(text).tolist()
            }
            
            # Add context-aware analysis if we have context
            if self.context_window:
                analysis["context"] = {
                    "conversation_flow": self._analyze_conversation_flow(),
                    "related_topics": self._find_related_topics(text),
                    "context_relevance": self._calculate_context_relevance(text)
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            # Return basic analysis if advanced processing fails
            return {
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "topics": ["general"],
                "sentiment": 0.0,
                "entities": {},
                "intent": {"intent": "unknown", "confidence": 0.0}
            }
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text using keyword extraction and Grok"""
        try:
            # Try using Grok for sophisticated topic extraction
            if self.grok_client.api_key:
                prompt = f"Extract the main topics from this text as a comma-separated list: {text}"
                response = self.grok_client.ask(prompt, temperature=0.3)
                if not response.startswith("[Grok API Error]"):
                    return [topic.strip() for topic in response.split(",")]
            
            # Fall back to keyword-based extraction
            # Remove common stop words and extract frequent terms
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "is", "are"}
            words = text.lower().split()
            keywords = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Count word frequencies
            from collections import Counter
            word_freq = Counter(keywords)
            
            # Return top 3 most frequent words as topics
            return [word for word, _ in word_freq.most_common(3)] or ["general"]
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return ["general"]
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (range -1 to 1)"""
        try:
            if self.initialized:
                # Use transformer pipeline for sentiment analysis
                result = self.sentiment_analyzer(text)[0]
                # Convert label and score to -1 to 1 range
                sentiment_value = result["score"]
                if result["label"] == "NEGATIVE":
                    sentiment_value = -sentiment_value
                
                # Handle neutral sentiment more strictly
                # Check for neutral words and phrases
                neutral_words = {"cloudy", "today", "weather", "is", "the", "a", "an", "this", "that"}
                words = set(text.lower().split())
                if len(words.intersection(neutral_words)) / len(words) > 0.5:
                    return 0.0
                if abs(sentiment_value) < 0.3:
                    return 0.0
                    
                return sentiment_value
            
            # Fall back to basic sentiment analysis
            positive_words = {"good", "great", "excellent", "amazing", "wonderful", "fantastic"}
            negative_words = {"bad", "terrible", "awful", "horrible", "poor", "wrong"}
            
            words = text.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            total = positive_count + negative_count
            if total == 0:
                return 0.0
            return (positive_count - negative_count) / total
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return 0.0
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities and other structured information from text"""
        entities = {}
        
        try:
            if self.initialized:
                # Use transformer pipeline for named entity recognition
                ner_results = self.ner_pipeline(text)
                
                # Group entities by type
                for ent in ner_results:
                    ent_type = ent["entity"].lower()
                    if ent_type not in entities:
                        entities[ent_type] = []
                    entities[ent_type].append(ent["word"])
            
            # Add rule-based entity extraction for specific patterns
            
            # Extract emails
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            if emails:
                entities["email"] = emails
            
            # Extract URLs
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
            if urls:
                entities["url"] = urls
            
            # Extract dates
            dates = re.findall(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}', text)
            if dates:
                entities["date"] = dates
            
            # Extract numbers
            numbers = re.findall(r'\b\d+\.?\d*\b', text)
            if numbers:
                entities["number"] = numbers
            
            # Extract app names
            app_patterns = {
                "browser": r'\b(chrome|firefox|safari|edge|brave)\b',
                "editor": r'\b(word|excel|powerpoint|notepad)\b',
                "media": r'\b(spotify|netflix|vlc|youtube)\b'
            }
            
            for app_type, pattern in app_patterns.items():
                matches = re.findall(pattern, text.lower())
                if matches:
                    entities[f"{app_type}_app"] = matches
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return {}
    
    def recognize_intent(self, text: str) -> Dict[str, Any]:
        """Enhanced intent recognition using both Grok and rule-based patterns"""
        try:
            # Try using Grok for sophisticated intent recognition
            if self.grok_client.api_key:
                grok_intent = self.grok_client.analyze_intent(text)
                # Map Grok intents to our expected intents
                intent_mapping = {
                    "get_news": "web_search",
                    "query": "web_search",
                    "search": "web_search",
                    "check_email": "email",
                    "read_email": "email",
                    "mail": "email"
                }
                intent = grok_intent["intent"]
                mapped_intent = intent_mapping.get(intent, intent)
                if mapped_intent != "unknown":
                    return {
                        "intent": mapped_intent,
                        "confidence": 0.9,
                        "parameters": grok_intent.get("parameters", {})
                    }
            
            # Fall back to enhanced rule-based intent recognition
            text = text.lower()
            
            # Define comprehensive intent patterns
            intent_patterns = {
                "launch_app": {
                    "patterns": ["open", "launch", "start", "run"],
                    "keywords": ["app", "application", "program", "software"]
                },
                "system_control": {
                    "patterns": ["system", "cpu", "memory", "disk", "status", "performance"],
                    "keywords": ["usage", "check", "monitor", "speed"]
                },
                "web_search": {
                    "patterns": ["search", "find", "look up", "google"],
                    "keywords": ["web", "internet", "online", "browser"]
                },
                "email": {
                    "patterns": ["email", "mail", "inbox", "message"],
                    "keywords": ["send", "read", "check", "compose"]
                },
                "media_control": {
                    "patterns": ["play", "pause", "stop", "volume"],
                    "keywords": ["music", "video", "audio", "sound"]
                },
                "file_management": {
                    "patterns": ["file", "folder", "directory", "document"],
                    "keywords": ["create", "delete", "move", "copy"]
                },
                "time_date": {
                    "patterns": ["time", "date", "schedule", "calendar"],
                    "keywords": ["what", "current", "now", "today"]
                }
            }
            
            # Calculate confidence scores for each intent with priority weights
            scores = {}
            priority_patterns = {
                "email": ["check email", "check mail", "read email", "read mail"],
                "web_search": ["search for", "look up", "find information"]
            }
            
            # First check priority patterns
            for intent, patterns in priority_patterns.items():
                if any(pattern in text for pattern in patterns):
                    return {
                        "intent": intent,
                        "confidence": 0.9,
                        "parameters": {}
                    }
            
            # Then check regular patterns
            for intent, data in intent_patterns.items():
                pattern_matches = sum(1 for p in data["patterns"] if p in text)
                keyword_matches = sum(1 for k in data["keywords"] if k in text)
                total_patterns = len(data["patterns"]) + len(data["keywords"])
                scores[intent] = (pattern_matches + keyword_matches) / total_patterns if total_patterns > 0 else 0
            
            # Find the intent with highest confidence
            max_confidence = max(scores.values()) if scores else 0
            if max_confidence > 0:
                detected_intent = max(scores.items(), key=lambda x: x[1])[0]
            else:
                detected_intent = "unknown"
            
            return {
                "intent": detected_intent,
                "confidence": max_confidence,
                "scores": scores
            }
            
        except Exception as e:
            logger.error(f"Error in intent recognition: {e}")
            return {"intent": "unknown", "confidence": 0.0}
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get vector embedding for text using BERT"""
        try:
            if self.initialized:
                # Tokenize and get BERT embedding
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                # Use mean pooling to get sentence embedding
                embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
                return embedding
            
            # Fall back to basic embedding if BERT not initialized
            return np.zeros(self.embedding_size)
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_size)
    
    def _update_context(self, text: str):
        """Update conversation context window"""
        self.context_window.append({
            "text": text,
            "timestamp": datetime.now(),
            "embedding": self.get_embedding(text)
        })
        
        # Maintain fixed context window size
        if len(self.context_window) > self.max_context_items:
            self.context_window.pop(0)
    
    def _analyze_conversation_flow(self) -> Dict[str, Any]:
        """Analyze the flow of conversation from context window"""
        if not self.context_window:
            return {"flow_type": "new", "topic_consistency": 0.0}
        
        # Calculate topic consistency
        embeddings = [item["embedding"] for item in self.context_window]
        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        return {
            "flow_type": "continuous" if avg_similarity > 0.5 else "branching",
            "topic_consistency": float(avg_similarity),
            "context_length": len(self.context_window)
        }
    
    def _find_related_topics(self, text: str) -> List[str]:
        """Find topics from context that are related to current text"""
        if not self.context_window:
            return []
        
        current_embedding = self.get_embedding(text)
        related_topics = []
        
        for item in self.context_window[:-1]:  # Exclude current text
            similarity = np.dot(current_embedding, item["embedding"]) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(item["embedding"])
            )
            if similarity > 0.7:  # High similarity threshold
                topics = self.extract_topics(item["text"])
                related_topics.extend(topics)
        
        return list(set(related_topics))  # Remove duplicates
    
    def _calculate_context_relevance(self, text: str) -> float:
        """Calculate how relevant the current context window is to new text"""
        if not self.context_window:
            return 0.0
        
        current_embedding = self.get_embedding(text)
        similarities = []
        
        for item in self.context_window:
            similarity = np.dot(current_embedding, item["embedding"]) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(item["embedding"])
            )
            similarities.append(similarity)
        
        return float(np.mean(similarities))

    def extract_features(self, text: str) -> np.ndarray:
        """Extract comprehensive feature vector for neural network input"""
        # Combine embedding with additional features
        embedding = self.get_embedding(text)
        
        # Get sentiment score
        sentiment = self.analyze_sentiment(text)
        
        # Get intent confidence scores
        intent_result = self.recognize_intent(text)
        intent_score = intent_result.get("confidence", 0.0)
        
        # Combine features
        features = np.concatenate([
            embedding,
            np.array([sentiment, intent_score])
        ])
        
        return features
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            emb1 = self.get_embedding(text1)
            emb2 = self.get_embedding(text2)
            # Cosine similarity
            return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def plan_task(self, command: str) -> Dict[str, Any]:
        """Plan execution steps for a given command using intent recognition"""
        try:
            # Analyze command intent
            intent_analysis = self.recognize_intent(command)
            intent = intent_analysis["intent"]
            
            # Extract entities
            entities = self.extract_entities(command)
            
            # Define task steps based on intent
            steps = []
            estimated_time = 0
            
            if intent == "launch_app" or "browser" in command.lower():
                app_name = next(iter(entities.get("browser_app", [])), None) or \
                          next(iter(entities.get("editor_app", [])), None) or \
                          next(iter(entities.get("media_app", [])), None)
                if app_name:
                    if "browser" in app_name or "chrome" in app_name.lower():
                        steps = [
                            {"action": "launch_browser", "params": {"browser": app_name}},
                            {"action": "verify_launch", "params": {"app_name": app_name}}
                        ]
                        # If command includes search, add search steps
                        if "search" in command.lower():
                            query = command.split("search for")[-1].strip() if "search for" in command.lower() else \
                                   command.split("search")[-1].strip()
                            steps.extend([
                                {"action": "navigate", "params": {"url": "https://www.google.com"}},
                                {"action": "search", "params": {"query": query}}
                            ])
                    else:
                        steps = [
                            {"action": "launch_app", "params": {"app_name": app_name}},
                            {"action": "verify_launch", "params": {"app_name": app_name}}
                        ]
                    estimated_time = 3
            
            elif intent == "web_search":
                query = command.split("search for")[-1].strip() if "search for" in command else command
                steps = [
                    {"action": "launch_browser", "params": {"browser": "default"}},
                    {"action": "navigate", "params": {"url": "https://www.google.com"}},
                    {"action": "search", "params": {"query": query}}
                ]
                estimated_time = 5
            
            elif intent == "system_control":
                steps = [
                    {"action": "check_system", "params": {
                        "check_cpu": True,
                        "check_memory": True,
                        "check_disk": True
                    }},
                    {"action": "generate_report", "params": {"format": "text"}}
                ]
                estimated_time = 2
            
            elif intent == "email":
                # Extract email address if present
                email_to = next(iter(entities.get("email", [])), None)
                steps = [
                    {"action": "launch_email", "params": {}},
                    {"action": "compose_email", "params": {
                        "to": email_to,
                        "subject": "New Email"
                    }}
                ]
                estimated_time = 3
            
            elif intent == "file_management":
                operation = "create" if "create" in command else \
                           "delete" if "delete" in command else \
                           "move" if "move" in command else "list"
                steps = [
                    {"action": "file_operation", "params": {
                        "operation": operation,
                        "path": "."  # Default to current directory
                    }}
                ]
                estimated_time = 1
            
            else:
                # Default to asking Grok for help
                steps = [
                    {"action": "grok_query", "params": {"query": command}}
                ]
                estimated_time = 2
            
            return {
                "steps": steps,
                "estimated_time": estimated_time,
                "intent": intent,
                "confidence": intent_analysis["confidence"],
                "entities": entities
            }
            
        except Exception as e:
            logger.error(f"Error planning task: {e}")
            return {
                "steps": [],
                "estimated_time": 0,
                "intent": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }

# Create global instances for backward compatibility
nlp_processor = NLPProcessor()
recognize_intent = nlp_processor.recognize_intent
extract_entities = nlp_processor.extract_entities
plan_task = nlp_processor.plan_task
