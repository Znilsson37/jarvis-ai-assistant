# Jarvis AI Assistant - Quick Start Guide

## Installation

1. **Prerequisites**
   - Python 3.8 or later
   - Windows operating system
   - Microphone for voice input
   - Speakers for voice output
   - Brave browser installed (for web automation)

2. **Quick Install**
   - Double-click `install.bat`
   - Follow the prompts
   - Wait for installation to complete

3. **Manual Installation**
   ```bash
   # Create virtual environment
   python -m venv venv
   venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt

   # Install browser automation
   playwright install

   # Install package
   pip install -e .
   ```

## Configuration

1. **API Keys**
   Edit `Config/config.json`:
   ```json
   {
     "eleven_labs_api_key": "your-key-here",
     "openai_api_key": "your-key-here",
     "grok_api_key": "your-key-here"
   }
   ```

2. **System Permissions**
   - Allow microphone access
   - Allow system control access
   - Allow browser automation

## Running Jarvis

1. **Start the Assistant**
   ```bash
   venv\Scripts\activate
   python main.py
   ```

2. **Basic Commands**
   - "Jarvis" (wake word)
   - "Jarvis, what time is it?"
   - "Jarvis, system status"
   - "Jarvis, open Chrome"
   - "Jarvis, search for..."
   - "Jarvis, check my email"

## Features

1. **Voice Control**
   - Natural language commands
   - High-quality text-to-speech
   - Wake word detection

2. **System Control**
   - Application management
   - System monitoring
   - Performance optimization
   - Diagnostics and repair

3. **Web Integration**
   - Brave browser automation
   - Email access
   - Web search
   - Information retrieval

4. **UI Features**
   - Sound-responsive visualization
   - System status display
   - Dark theme interface

## Troubleshooting

1. **Voice Recognition Issues**
   - Check microphone settings
   - Verify wake word ("Jarvis")
   - Speak clearly and at moderate speed

2. **System Control Issues**
   - Run as administrator
   - Check permissions
   - Verify system compatibility

3. **Browser Issues**
   - Ensure Brave is installed
   - Check internet connection
   - Verify browser automation setup

## Support

- Check README.md for detailed documentation
- Review logs in case of errors
- Submit issues on GitHub

## Development

1. **Running Tests**
   ```bash
   python -m pytest tests/test_integration.py -v
   ```

2. **Adding Features**
   - Create new modules in `modules/`
   - Update main.py
   - Add tests
   - Update documentation

## Security Notes

- API keys are stored locally
- System access is protected
- Browser automation is sandboxed
- Voice data is processed locally when possible

## Updates

- Check for updates regularly
- Pull latest from repository
- Run install.bat to update dependencies

For more detailed information, see README.md
