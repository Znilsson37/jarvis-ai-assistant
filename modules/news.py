import os
import json
import datetime
import requests
from typing import Dict, List, Optional
import logging
from modules.config import config

logger = logging.getLogger(__name__)

class NewsController:
    def __init__(self):
        self.api_key = config.get("grok_api_key")
        self.api_url = "https://api.x.ai/v1/chat/completions"
        self.model = "grok-3-latest"
        
    def _make_request(self, messages: List[Dict], search_params: Dict) -> Dict:
        """Make a request to the Grok API"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "messages": messages,
                "search_parameters": search_params,
                "model": self.model
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return {"error": f"Failed to fetch news: {str(e)}"}
            
    def get_daily_brief(self, categories: Optional[List[str]] = None) -> Dict:
        """Get a daily news briefing"""
        try:
            # Default categories if none provided
            if not categories:
                categories = ["world", "technology", "business", "science"]
            
            # Create category-specific prompts
            category_prompts = []
            for category in categories:
                category_prompts.append(
                    f"Provide a brief summary of today's top {category} news. "
                    "Focus on the most significant developments."
                )
            
            combined_prompt = " ".join(category_prompts)
            
            messages = [{
                "role": "user",
                "content": combined_prompt
            }]
            
            search_params = {
                "mode": "on",  # Force search to be enabled
                "sources": [
                    {"type": "news"},
                    {"type": "web"},
                    {"type": "x", "x_handles": ["BBCBreaking", "CNN", "Reuters"]}
                ],
                "return_citations": True,
                "max_search_results": 20
            }
            
            response = self._make_request(messages, search_params)
            
            if "error" in response:
                return response
                
            # Extract the content and citations
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            citations = response.get("citations", [])
            
            return {
                "content": content,
                "citations": citations,
                "timestamp": datetime.datetime.now().isoformat(),
                "categories": categories
            }
            
        except Exception as e:
            logger.error(f"Error getting daily brief: {str(e)}")
            return {"error": f"Failed to generate daily brief: {str(e)}"}
    
    def search_news(self, query: str, days_back: int = 1, country: Optional[str] = None) -> Dict:
        """Search for specific news topics"""
        try:
            from_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
            
            messages = [{
                "role": "user",
                "content": f"Find and summarize news about: {query}"
            }]
            
            search_params = {
                "mode": "on",
                "from_date": from_date,
                "sources": [
                    {
                        "type": "news",
                        "country": country if country else None
                    },
                    {"type": "web"},
                    {"type": "x"}
                ],
                "return_citations": True,
                "max_search_results": 10
            }
            
            # Remove None values from sources
            search_params["sources"] = [
                {k: v for k, v in source.items() if v is not None}
                for source in search_params["sources"]
            ]
            
            response = self._make_request(messages, search_params)
            
            if "error" in response:
                return response
                
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            citations = response.get("citations", [])
            
            return {
                "content": content,
                "citations": citations,
                "query": query,
                "from_date": from_date,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error searching news: {str(e)}")
            return {"error": f"Failed to search news: {str(e)}"}
    
    def get_tech_updates(self) -> Dict:
        """Get latest technology news and updates"""
        try:
            messages = [{
                "role": "user",
                "content": (
                    "Provide a summary of the latest significant technology news "
                    "and developments, focusing on major announcements, breakthroughs, "
                    "and industry trends."
                )
            }]
            
            search_params = {
                "mode": "on",
                "sources": [
                    {"type": "news"},
                    {"type": "web"},
                    {
                        "type": "x",
                        "x_handles": [
                            "TechCrunch", "WIRED", "TheVerge", "xAI", 
                            "OpenAI", "Google", "Microsoft"
                        ]
                    }
                ],
                "return_citations": True,
                "max_search_results": 15
            }
            
            response = self._make_request(messages, search_params)
            
            if "error" in response:
                return response
                
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            citations = response.get("citations", [])
            
            return {
                "content": content,
                "citations": citations,
                "category": "technology",
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting tech updates: {str(e)}")
            return {"error": f"Failed to get tech updates: {str(e)}"}
    
    def get_local_news(self, country_code: str) -> Dict:
        """Get local news for a specific country"""
        try:
            messages = [{
                "role": "user",
                "content": f"Provide a summary of important local news and events in {country_code}"
            }]
            
            search_params = {
                "mode": "on",
                "sources": [
                    {"type": "news", "country": country_code},
                    {"type": "web", "country": country_code}
                ],
                "return_citations": True,
                "max_search_results": 10
            }
            
            response = self._make_request(messages, search_params)
            
            if "error" in response:
                return response
                
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            citations = response.get("citations", [])
            
            return {
                "content": content,
                "citations": citations,
                "country": country_code,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting local news: {str(e)}")
            return {"error": f"Failed to get local news: {str(e)}"}

# Create global news controller instance
news_controller = NewsController()
