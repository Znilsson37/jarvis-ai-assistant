import asyncio
import sys
import os
import time
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add current directory to path to make imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.config import config
from modules.speech import speech
from modules import nlp
from modules.system_control import system
from modules.news import news_controller
from modules import browser_control
from modules.system_diagnostics import diagnostics
from jarvis_ui import JarvisUIIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JarvisAssistant:
    def __init__(self):
        self.running = False
        self.wake_phrase = "jarvis"
        self.setup_complete = False
        self.browser = None
        self.ui = None
        
    async def setup(self):
        """Initialize all necessary components"""
        if self.setup_complete:
            return
            
        logger.info("Setting up Jarvis Assistant...")
        
        try:
            # Initialize speech components
            speech.load_models()
            
            
            # Initialize browser controller
            await browser_control.initialize()
            self.browser = browser_control
            
            # Initialize UI
            self.ui = JarvisUIIntegration()
            threading.Thread(target=self.ui.start_ui, daemon=True).start()
            time.sleep(1)  # Give UI time to initialize
            
            # Check for API keys
            self._check_api_keys()
            
            # Run initial system diagnostics
            await self._run_initial_diagnostics()
            
            self.setup_complete = True
            logger.info("Setup complete!")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise
    
    def _check_api_keys(self):
        """Check for required API keys"""
        required_keys = {
            "openai_api_key": "OpenAI",
            "eleven_labs_api_key": "Eleven Labs",
            "grok_api_key": "Grok"
        }
        
        for key, service in required_keys.items():
            if not config.get(key):
                logger.warning(f"No {service} API key found. Some features will be limited.")
    
    async def _run_initial_diagnostics(self):
        """Run initial system diagnostics"""
        try:
            logger.info("Running initial system diagnostics...")
            results = diagnostics.run_full_diagnostics()
            
            # Check for critical issues
            critical_issues = [
                component for component, data in results.items()
                if isinstance(data, dict) and data.get("status") == "critical"
            ]
            
            if critical_issues:
                logger.warning(f"Critical issues found in: {', '.join(critical_issues)}")
                # Attempt automatic repair
                await self._repair_critical_issues(critical_issues)
            
        except Exception as e:
            logger.error(f"Diagnostics failed: {e}")
    
    async def _repair_critical_issues(self, issues):
        """Attempt to repair critical system issues"""
        try:
            logger.info(f"Attempting to repair issues: {issues}")
            repair_result = diagnostics.repair_system(issues)
            
            if repair_result["status"] == "completed":
                logger.info("Repairs completed successfully")
            else:
                logger.warning("Some repairs could not be completed")
                
        except Exception as e:
            logger.error(f"Repair attempt failed: {e}")
    
    async def process_command(self, command_text: str):
        """Process a voice command"""
        logger.info(f"Processing command: '{command_text}'")
        
        # Check for wake phrase
        if self.wake_phrase not in command_text.lower():
            return
        
        # Remove wake phrase
        command_text = command_text.lower().replace(self.wake_phrase, "").strip()
        if not command_text:
            response = "Yes, how can I help you?"
            await self._speak(response)
            return
        
        # Recognize intent
        intent_result = nlp.recognize_intent(command_text)
        intent = intent_result.get("intent", "unknown")
        
        try:
            # Process based on intent
            if intent == "system_control":
                await self._handle_system_control(command_text)
            elif intent == "browser":
                await self._handle_browser_command(command_text)
            elif intent == "diagnostics":
                await self._handle_diagnostics(command_text)
            elif intent == "email":
                await self._handle_email(command_text)
            else:
                await self._handle_general_query(command_text)
                
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            await self._speak("I encountered an error processing that command")
    
    async def _handle_system_control(self, command: str):
        """Handle system control commands"""
        if "status" in command:
            info = system.get_system_info()
            response = (
                f"CPU usage is {info['cpu_percent']}%, "
                f"Memory usage is {info['memory_percent']}%, "
                f"Disk usage is {info['disk_percent']}%"
            )
            if self.ui:
                self.ui.update_system_info(
                    info['cpu_percent'],
                    info['memory_percent'],
                    info['disk_percent']
                )
        elif "launch" in command or "open" in command:
            app_name = nlp.extract_app_name(command)
            if app_name:
                if app_name.lower() == "gmail":
                    if self.browser:
                        await self.browser.navigate("https://mail.google.com")
                        response = "Opening Gmail in browser."
                    else:
                        response = "Browser control is not initialized."
                elif app_name.lower() == "zoom":
                    response = system.launch_app("zoom")
                else:
                    response = system.launch_app(app_name)
            else:
                response = "Which application would you like to open?"
        else:
            response = "I'm not sure what system control you'd like me to perform"
            
        await self._speak(response)
    
    async def _handle_browser_command(self, command: str):
        """Handle browser-related commands"""
        if not self.browser:
            await self._speak("Browser control is not initialized")
            return
            
        if "search" in command:
            query = command.replace("search", "").strip()
            result = await self.browser.search(query)
            if result["status"] == "success":
                response = f"Here are the search results for {query}"
            else:
                response = "I had trouble performing that search"
        else:
            response = "I'm not sure what browser action you'd like me to perform"
            
        await self._speak(response)
    
    async def _handle_diagnostics(self, command: str):
        """Handle system diagnostic commands"""
        if "run" in command or "check" in command:
            results = diagnostics.run_full_diagnostics()
            issues = [
                component for component, data in results.items()
                if isinstance(data, dict) and data.get("status") in ["warning", "critical"]
            ]
            
            if issues:
                response = f"I found issues with: {', '.join(issues)}. Would you like me to attempt repairs?"
            else:
                response = "All systems are functioning normally"
        else:
            response = "What kind of diagnostic would you like me to run?"
            
        await self._speak(response)
    
    async def _handle_email(self, command: str):
        """Handle email-related commands"""
        if not self.browser:
            await self._speak("Email functionality is not available without browser control")
            return
            
        if "check" in command:
            result = await self.browser.check_email()
            if result["status"] == "success":
                unread = sum(1 for email in result["emails"] if email["unread"])
                response = f"You have {unread} unread emails"
            else:
                response = "I couldn't access your email"
        else:
            response = "What would you like me to do with email?"
            
        await self._speak(response)
    
    async def _handle_general_query(self, command: str):
        """Handle general queries using NLP"""
        try:
            # Use Grok client from nlp module
            if nlp.client and nlp.client.api_key:
                # Special case for news commands
                if "news" in command.lower():
                    news_result = news_controller.get_daily_brief()
                    if "error" in news_result:
                        response = f"Failed to fetch news: {news_result['error']}"
                    else:
                        response = news_result.get("content", "No news available.")
                else:
                    response = nlp.get_response_for_intent(nlp.recognize_intent(command))
            else:
                response = "I need a Grok API key to answer complex queries"
            
            await self._speak(response)
        except Exception as e:
            logger.error(f"Error handling general query: {e}")
            await self._speak("I encountered an error processing that query")
    
    async def _speak(self, text: str):
        """Speak response using text-to-speech"""
        if self.ui:
            self.ui.on_speaking(text)
        speech.speak(text)
        if self.ui:
            self.ui.on_finish_speaking()
    
    async def listen_once(self):
        """Listen for a single command"""
        logger.info("Listening...")
        if self.ui:
            self.ui.on_listening()
        speech.speak("Listening...")
        
        # Record audio
        audio_file = speech.record_audio(5)
        
        # Transcribe
        command_text = speech.transcribe(audio_file)
        
        # Process command
        if command_text and self.ui:
            self.ui.on_processing_command(command_text)
        await self.process_command(command_text)
    
    async def run_interactive(self):
        """Run in interactive mode, listening for commands"""
        await self.setup()
        
        self.running = True
        await self._speak("Jarvis is online")
        
        try:
            while self.running:
                await self.listen_once()
                await asyncio.sleep(1)  # Short delay between listening sessions
        except KeyboardInterrupt:
            self.running = False
            await self._speak("Jarvis shutting down")
            logger.info("Shutting down...")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        # Clean up browser
        if self.browser:
            try:
                await self.browser.cleanup()
            except Exception as e:
                logger.error(f"Error during browser cleanup: {e}")
        
        # Clean up UI
        if self.ui:
            try:
                self.ui.root.quit()
            except Exception as e:
                logger.error(f"Error during UI cleanup: {e}")
            
        # Clean up speech
        try:
            speech.cleanup()
        except Exception as e:
            logger.error(f"Error during speech cleanup: {e}")
    
    def run(self):
        """Main entry point"""
        asyncio.run(self.run_interactive())

if __name__ == "__main__":
    jarvis = JarvisAssistant()
    jarvis.run()
