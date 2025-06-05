import asyncio
import os
import time
import threading
import signal
import sys
from pathlib import Path
import traceback
import logging

from modules.speech import speech
from modules import nlp
from modules import system_control as system
from modules.security import security
from modules.config import config
from modules import keyboard_control
from modules import camera
from modules import office_control
from jarvis_ui import JarvisUIIntegration

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JarvisAssistant:
    def __init__(self):
        self.running = False
        self.wake_phrase = "jarvis"
        self.setup_complete = False
        self.context_memory = []
        self.ui_integration = JarvisUIIntegration()
        self.continuous_listening = True
        self.voice_threshold = 500  # Adjust based on your microphone sensitivity
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def setup(self):
        """Enhanced setup with better error handling"""
        if not self.setup_complete:
            try:
                print("\nSetting up Jarvis Assistant...\n")
                
                # Load speech models with timeout
                logger.info("Loading speech models...")
                speech.load_models()
                logger.info("Speech models loaded successfully")

                # Check API keys
                api_key = security.get_api_key()
                grok_key = config.get("grok_api_key")
                
                if not api_key and not config.get("openai_api_key") and not grok_key:
                    print("\nNo API keys found. Advanced features will be limited.")
                    print("You can set your API keys in the config.json file.")
                elif grok_key:
                    logger.info("Grok API key found - advanced features enabled")

                self.setup_complete = True
                print("\nSetup complete!")
                
            except Exception as e:
                logger.error(f"Setup failed: {str(e)}")
                logger.error(traceback.format_exc())
                raise

    def is_wake_word_detected(self, command_text):
        """Improved wake word detection"""
        if not command_text:
            return False
        
        command_lower = command_text.lower().strip()
        wake_variations = [
            self.wake_phrase,
            "hey " + self.wake_phrase,
            "ok " + self.wake_phrase,
            "computer",
            "assistant"
        ]
        
        return any(wake in command_lower for wake in wake_variations)

    async def process_command(self, command_text):
        """Enhanced command processing with better error handling"""
        if not command_text or not command_text.strip():
            return

        try:
            logger.info(f"Processing command: '{command_text}'")
            self.ui_integration.on_processing_command(command_text)
            self.context_memory.append(command_text)
            
            # Keep context memory manageable
            if len(self.context_memory) > 10:
                self.context_memory = self.context_memory[-10:]

            # Check for wake word
            if not self.is_wake_word_detected(command_text):
                # If no wake word, just store for context but don't process
                return

            # Remove wake phrase and clean command
            command_cleaned = self.clean_command(command_text)
            
            if not command_cleaned:
                response = "Yes, how can I help you?"
                self.ui_integration.on_speaking(response)
                speech.speak(response)
                return

            # Get intent with timeout
            try:
                intent_result = await asyncio.wait_for(
                    asyncio.to_thread(nlp.recognize_intent, command_cleaned), 
                    timeout=10.0
                )
                logger.info(f"Intent Result: {intent_result}")
            except asyncio.TimeoutError:
                logger.warning("Intent recognition timed out, using fallback")
                intent_result = nlp.fallback_intent_parser(command_cleaned)
            except Exception as e:
                logger.error(f"Intent recognition failed: {str(e)}")
                intent_result = {"intent": "unknown"}

            # Process the intent
            await self.execute_intent(intent_result, command_cleaned)

        except Exception as e:
            logger.error(f"Error processing command: {str(e)}")
            logger.error(traceback.format_exc())
            error_msg = "I encountered an error processing your request. Please try again."
            self.ui_integration.on_speaking(error_msg)
            speech.speak(error_msg)
        finally:
            self.ui_integration.on_finish_speaking()

    def clean_command(self, command_text):
        """Clean and normalize the command text"""
        command_lower = command_text.lower()
        
        # Remove various wake phrases
        wake_patterns = [
            "hey jarvis", "ok jarvis", "jarvis", 
            "hey computer", "ok computer", "computer",
            "hey assistant", "ok assistant", "assistant"
        ]
        
        for pattern in wake_patterns:
            if command_lower.startswith(pattern):
                command_lower = command_lower[len(pattern):].strip()
                break
            elif pattern in command_lower:
                command_lower = command_lower.replace(pattern, "").strip()
                break
        
        return command_lower

    async def execute_intent(self, intent_result, command_text):
        """Execute the recognized intent"""
        intent = intent_result.get("intent", "unknown")
        entities = intent_result if isinstance(intent_result, dict) else {}

        try:
            if intent == "launch_app":
                await self.handle_launch_app(entities, command_text)
            elif intent == "file_management":
                await self.handle_file_management(command_text)
            elif intent == "system_control":
                await self.handle_system_control()
            elif intent == "get_news":
                await self.handle_get_news()
            elif intent == "query":
                await self.handle_query(command_text)
            elif intent == "type_text":
                await self.handle_type_text(entities, command_text)
            elif intent == "run_command":
                await self.handle_run_command(entities, command_text)
            elif intent == "take_photo":
                await self.handle_take_photo()
            elif intent == "open_word":
                await self.handle_open_word()
            elif intent == "open_excel":
                await self.handle_open_excel()
            elif intent == "create_doc":
                await self.handle_create_doc(entities, command_text)
            else:
                await self.handle_unknown_intent(command_text)
                
        except Exception as e:
            logger.error(f"Error executing intent {intent}: {str(e)}")
            error_msg = f"I had trouble with that {intent.replace('_', ' ')} request."
            self.speak_and_log(error_msg)

    async def handle_launch_app(self, entities, command_text):
        """Handle app launching with improved app name extraction"""
        app_name = entities.get("app_name", "")
        
        if not app_name:
            # Try to extract app name from command
            import re
            patterns = [
                r"open (.+?)(?:\s|$)",
                r"launch (.+?)(?:\s|$)",
                r"start (.+?)(?:\s|$)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, command_text, re.IGNORECASE)
                if match:
                    app_name = match.group(1).strip()
                    break
        
        if app_name:
            result = system.launch_app(app_name)
            self.speak_and_log(result)
        else:
            msg = "Which application would you like me to open?"
            self.speak_and_log(msg)

    async def handle_file_management(self, command_text):
        """Handle file management operations"""
        if "organize" in command_text and "download" in command_text:
            downloads_path = Path.home() / "Downloads"
            result = system.organize_files(downloads_path)
            self.speak_and_log(result)
        else:
            msg = "I'm not sure what file operation you want me to perform."
            self.speak_and_log(msg)

    async def handle_system_control(self):
        """Handle system control requests"""
        info = system.system_status()
        self.speak_and_log(info)

    async def handle_get_news(self):
        """Handle news requests"""
        try:
            from modules import news
            result = await asyncio.to_thread(news.fetch_latest_news)
            self.speak_and_log(result)
        except Exception as e:
            logger.error(f"News fetch error: {str(e)}")
            self.speak_and_log("I'm having trouble fetching the news right now.")

    async def handle_query(self, command_text):
        """Handle general queries"""
        if nlp.client:
            try:
                # Build context-aware message
                context_prompt = (
                    "Based on previous context: " + 
                    "; ".join(self.context_memory[-5:]) + 
                    f"\nUser: {command_text}"
                )
                result = await asyncio.to_thread(nlp.client.ask, context_prompt)
                self.speak_and_log(result)
            except Exception as e:
                logger.error(f"Query processing error: {str(e)}")
                self.speak_and_log("I'm having trouble processing that query right now.")
        else:
            msg = "I need an API key to answer complex queries."
            self.speak_and_log(msg)

    async def handle_type_text(self, entities, command_text):
        """Handle text typing requests"""
        text = entities.get("text", "")
        if not text:
            # Extract text from command
            import re
            match = re.search(r"type(?:\s+out)?(?:\s+this)?(?:\s+text)?:?\s*(.+)", command_text, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
            else:
                text = command_text.replace("type", "").strip()
        
        if text:
            keyboard_control.type_text(text)
            self.speak_and_log(f"Typing: {text}")
        else:
            self.speak_and_log("What would you like me to type?")

    async def handle_run_command(self, entities, command_text):
        """Handle run command requests"""
        cmd = entities.get("command", "")
        if not cmd:
            # Extract command from text
            import re
            match = re.search(r"run\s+(.+)", command_text, re.IGNORECASE)
            if match:
                cmd = match.group(1).strip()
        
        if cmd:
            keyboard_control.execute_run_command(cmd)
            self.speak_and_log(f"Running: {cmd}")
        else:
            self.speak_and_log("What command should I run?")

    async def handle_take_photo(self):
        """Handle photo capture requests"""
        result = await asyncio.to_thread(camera.capture_image)
        self.speak_and_log(result)

    async def handle_open_word(self):
        """Handle Word opening requests"""
        result = await asyncio.to_thread(office_control.open_word)
        self.speak_and_log(result)

    async def handle_open_excel(self):
        """Handle Excel opening requests"""
        result = await asyncio.to_thread(office_control.open_excel)
        self.speak_and_log(result)

    async def handle_create_doc(self, entities, command_text):
        """Handle document creation requests"""
        text = entities.get("text", "")
        if not text:
            # Extract text from command
            import re
            match = re.search(r"create.*?document.*?(?:with|that says|saying)\s*(.+)", command_text, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
        
        result = await asyncio.to_thread(office_control.create_word_doc_with_text, text)
        self.speak_and_log(result)

    async def handle_unknown_intent(self, command_text):
        """Handle unknown intents"""
        responses = [
            "I'm not sure how to help with that yet.",
            "Could you rephrase that request?",
            "I don't understand that command. Can you try saying it differently?",
            "That's not something I can do right now. What else can I help with?"
        ]
        import random
        msg = random.choice(responses)
        self.speak_and_log(msg)

    def speak_and_log(self, message):
        """Utility method to speak and log messages"""
        logger.info(f"Response: {message}")
        self.ui_integration.on_speaking(message)
        speech.speak(message)

    async def listen_once(self):
        """Improved listening with better error handling"""
        try:
            self.ui_integration.on_listening()
            logger.info("Starting to listen...")
            
            # Record audio with timeout
            audio_file = await asyncio.to_thread(speech.record_audio, 5)
            
            if not audio_file or not os.path.exists(audio_file):
                logger.warning("No audio file created")
                return
            
            logger.info("Transcribing audio...")
            # Transcribe with timeout
            command_text = await asyncio.to_thread(speech.transcribe, audio_file)
            
            if command_text:
                logger.info(f"Transcription: {command_text}")
                await self.process_command(command_text)
            else:
                logger.info("No speech detected")
                
        except Exception as e:
            logger.error(f"Error in listen_once: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            # Clean up audio file
            try:
                if 'audio_file' in locals() and audio_file and os.path.exists(audio_file):
                    os.remove(audio_file)
            except:
                pass

    async def run_interactive(self):
        """Main interactive loop with enhanced error handling"""
        try:
            await self.setup()

            self.running = True
            greeting = "Good day! Jarvis is online and ready to assist. How can I help you?"
            self.ui_integration.on_speaking(greeting)
            speech.speak(greeting)
            self.ui_integration.on_finish_speaking()

            logger.info("Starting interactive mode...")

            while self.running:
                try:
                    await self.listen_once()
                    # Small delay to prevent overwhelming the system
                    await asyncio.sleep(0.5)
                    
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    await asyncio.sleep(2)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Fatal error in interactive mode: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown"""
        if self.running:
            logger.info("Shutting down Jarvis...")
            self.running = False
            
            try:
                speech.speak("Jarvis shutting down. Goodbye!")
                time.sleep(2)  # Give time for speech to complete
            except:
                pass  # Don't let speech errors prevent shutdown
            
            print("\nJarvis shutdown complete.")

    def run(self):
        """Main run method with better error handling"""
        try:
            # Start UI in separate thread
            ui_thread = threading.Thread(target=self.ui_integration.start_ui)
            ui_thread.daemon = True
            ui_thread.start()

            # Give UI time to start
            time.sleep(2)
            
            # Run the main interactive loop
            asyncio.run(self.run_interactive())
            
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Fatal application error: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            self.shutdown()

if __name__ == "__main__":
    jarvis = JarvisAssistant()
    jarvis.run()