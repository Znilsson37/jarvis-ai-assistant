import requests
import json
import time
from modules.config import config

class GrokClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or config.get("grok_api_key")
        self.endpoint = "https://api.x.ai/v1/chat/completions"
        self.session = requests.Session()
        self.session.timeout = 30
        self.max_retries = 3
        self.retry_delay = 1

    def _make_request(self, payload, attempt=1):
        """Make request with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = self.session.post(
                self.endpoint, 
                headers=headers, 
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)
                return self._make_request(payload, attempt + 1)
            else:
                raise e

    def ask(self, prompt, model="grok-3-latest", max_tokens=1000, temperature=0.7, search_parameters=None):
        """Ask Grok a question using the chat completions format"""
        if not self.api_key:
            return "[Grok API Error] No API key provided"

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI-powered desktop personal assistant designed to enhance user productivity, streamline tasks, and provide helpful, accurate, and context-aware responses. Your name is JARVIS, and your primary goal is to assist users with a wide range of tasks while maintaining a professional, friendly, and approachable tone Your ultimate purpose is to make the user day easier, more organized, and more productive Always strive to understand the users intent, deliver value in every interaction, and maintain a balance between helpfulness and respect for their autonomy. If unsure about a request, seek clarification to ensure you meet the user’s needs effectively."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        if search_parameters:
            payload["search_parameters"] = search_parameters

        try:
            response_data = self._make_request(payload)
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"].strip()
            else:
                return "[Grok API Error] No response content received"
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return "[Grok API Error] Invalid API key"
            elif e.response.status_code == 429:
                return "[Grok API Error] Rate limit exceeded. Please try again later."
            elif e.response.status_code == 503:
                return "[Grok API Error] Service temporarily unavailable"
            else:
                return f"[Grok API Error] HTTP {e.response.status_code}: {e.response.text}"
        except requests.exceptions.ConnectionError:
            return "[Grok API Error] Connection failed. Please check your internet connection."
        except requests.exceptions.Timeout:
            return "[Grok API Error] Request timed out. Please try again."
        except requests.exceptions.RequestException as e:
            return f"[Grok API Error] {str(e)}"
        except json.JSONDecodeError:
            return "[Grok API Error] Invalid JSON response received"
        except Exception as e:
            return f"[Grok API Error] Unexpected error: {str(e)}"

    def chat(self, messages, model="grok-3-latest", max_tokens=1000, temperature=0.7, search_parameters=None):
        """Have a conversation with Grok using a list of messages"""
        if not self.api_key:
            return "[Grok API Error] No API key provided"

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        if search_parameters:
            payload["search_parameters"] = search_parameters

        try:
            response_data = self._make_request(payload)
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"].strip()
            else:
                return "[Grok API Error] No response content received"
                
        except Exception as e:
            return f"[Grok API Error] {str(e)}"

    def get_news(self, query="latest news", search_mode="auto", return_citations=True, from_date=None, to_date=None, max_search_results=20, sources=None):
        """Get news using Grok with live search capability"""
        if not self.api_key:
            return "I need an API key to fetch news."

        search_parameters = {
            "mode": search_mode,
            "return_citations": return_citations,
            "max_search_results": max_search_results
        }
        if from_date:
            search_parameters["from_date"] = from_date
        if to_date:
            search_parameters["to_date"] = to_date
        if sources:
            search_parameters["sources"] = sources

        payload = {
            "model": "grok-3-latest",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a news assistant. Provide current, accurate news information."
                },
                {
                    "role": "user",
                    "content": f"What are the {query}? Please provide current headlines and brief summaries."
                }
            ],
            "max_tokens": 800,
            "temperature": 0.3,
            "search_parameters": search_parameters
        }

        try:
            response_data = self._make_request(payload)
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"].strip()
            else:
                return "I'm unable to fetch the news at the moment."
                
        except Exception as e:
            return f"I couldn't fetch the news right now: {str(e)}"

    def analyze_intent(self, command):
        """Analyze user command intent"""
        prompt = f"""
        Analyze the following voice command and return a JSON object with the intent and extracted parameters.
        
        Available intents:
        - launch_app: Opening applications
        - file_management: File/folder operations
        - system_control: System information/control
        - get_news: News requests
        - query: General questions
        - type_text: Text input requests
        - run_command: System commands
        - take_photo: Camera operations
        - open_word: Microsoft Word
        - open_excel: Microsoft Excel
        - create_doc: Document creation
        - window_control: Window management
        - web_search: Web searches
        - time_date: Time/date requests
        - weather: Weather information
        - reminder: Setting reminders
        - calculation: Math operations
        - unknown: Unrecognized commands
        
        Command: "{command}"
        
        Return only valid JSON in this exact format:
        {{"intent": "intent_name", "parameters": {{"key": "value"}}}}
        """
        
        try:
            response = self.ask(prompt, temperature=0.1, max_tokens=200)
            # Try to extract JSON from the response
            if response.startswith("[Grok API Error]"):
                return {"intent": "unknown", "parameters": {}}
            
            # Find JSON in the response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                return {"intent": "unknown", "parameters": {}}
                
        except json.JSONDecodeError:
            return {"intent": "unknown", "parameters": {}}
        except Exception:
            return {"intent": "unknown", "parameters": {}}

    def summarize_text(self, text, max_length=100):
        """Summarize given text"""
        prompt = f"Summarize the following text in {max_length} words or less:\n\n{text}"
        return self.ask(prompt, max_tokens=max_length * 2)

    def translate_text(self, text, target_language="English"):
        """Translate text to target language"""
        prompt = f"Translate the following text to {target_language}:\n\n{text}"
        return self.ask(prompt, max_tokens=500)

    def get_model_info(self):
        """Get information about available models"""
        return {
            "available_models": ["mixtral-8x7b-32768", "llama2-70b-4096"],
            "default_model": "mixtral-8x7b-32768",
            "features": ["chat", "live_search", "code_generation", "analysis"]
        }

    def test_connection(self):
        """Test if the API connection is working"""
        try:
            response = self.ask("Hello, can you respond with 'Connection successful'?", max_tokens=10)
            if "Connection successful" in response or not response.startswith("[Grok API Error]"):
                return True
            return False
        except:
            return False
