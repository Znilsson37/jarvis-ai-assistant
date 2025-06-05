# Speech Recognition Enhancement Guide

## Overview

The Jarvis speech module has been significantly enhanced with comprehensive speech recognition capabilities using OpenAI's Whisper model, along with improved audio recording and voice activity detection.

## New Features

### 1. Speech Recognition with Whisper
- **Real-time transcription** using OpenAI Whisper
- **Multiple model sizes** supported (tiny, base, small, medium, large)
- **High accuracy** speech-to-text conversion
- **Multi-language support** (configurable)

### 2. Advanced Audio Recording
- **Voice Activity Detection (VAD)** for automatic recording stop
- **Multiple audio backends** (PyAudio and SoundDevice)
- **Configurable recording parameters** (duration, sample rate, etc.)
- **Automatic silence detection** to optimize recording length

### 3. Audio Analysis
- **Speech detection** in audio streams
- **Basic emotion analysis** from audio characteristics
- **Language detection** (placeholder for future enhancement)
- **Speaker recognition** (placeholder for future enhancement)

### 4. Enhanced TTS Integration
- **Dual TTS engine support** (Chatterbox and Eleven Labs)
- **Dynamic engine switching** at runtime
- **Voice cloning capabilities** with reference audio
- **Batch text-to-speech processing**

## Configuration

### Whisper Model Configuration
Edit `Config/config.json` to configure the Whisper model:

```json
{
    "whisper_model": "base",  // Options: tiny, base, small, medium, large
    "tts": {
        "default_engine": "chatterbox",  // or "eleven_labs"
        "chatterbox": {
            "device": "cuda",  // or "cpu"
            "output_dir": "audio_output",
            "default_exaggeration": 0.5,
            "default_cfg_weight": 0.5
        },
        "eleven_labs": {
            "default_voice_id": null,
            "default_stability": 0.5,
            "default_similarity_boost": 0.5
        }
    }
}
```

### Model Size Recommendations
- **tiny**: Fastest, lowest accuracy (~39 MB)
- **base**: Good balance of speed and accuracy (~74 MB) - **Recommended**
- **small**: Better accuracy, slower (~244 MB)
- **medium**: High accuracy (~769 MB)
- **large**: Best accuracy, slowest (~1550 MB)

## Usage Examples

### Basic Speech Recognition

```python
from modules import speech

# Load speech recognition models
speech.load_models()

# Record audio and transcribe
audio_file = speech.record_audio(duration=5.0, auto_stop=True)
if audio_file:
    transcript = speech.transcribe(audio_file)
    print(f"You said: {transcript}")
```

### Advanced Audio Analysis

```python
import numpy as np
from modules import speech

# Analyze audio data
audio_data = np.random.randn(16000).astype(np.float32)  # 1 second of audio
analysis = speech.analyze_audio(audio_data)

print(f"Transcript: {analysis['transcript']}")
print(f"Confidence: {analysis['confidence']}")
print(f"Emotion: {analysis['emotion']}")
print(f"Has Speech: {analysis['has_speech']}")
```

### TTS Engine Management

```python
from modules import speech

# Check current engine
current_engine = speech.speech_processor.tts_engine
print(f"Current TTS engine: {current_engine}")

# Switch engines
if speech.set_tts_engine("eleven_labs"):
    print("Switched to Eleven Labs")
    
# Generate speech
audio_file = speech.speak("Hello, this is a test of the speech system.")
if audio_file:
    print(f"Audio generated: {audio_file}")
```

### Voice Activity Detection

```python
import numpy as np
from modules import speech

# Test voice activity detection
audio_chunk = np.random.randn(1024).astype(np.float32) * 0.1
has_voice = speech.detect_speech(audio_chunk)
print(f"Voice detected: {has_voice}")
```

## API Reference

### Core Functions

#### `load_models()`
Load speech recognition models (Whisper).

#### `record_audio(duration=5.0, auto_stop=True)`
Record audio from microphone.
- **duration**: Maximum recording duration in seconds
- **auto_stop**: Whether to automatically stop on silence
- **Returns**: Path to recorded audio file or None

#### `transcribe(audio_file)`
Transcribe audio file to text using Whisper.
- **audio_file**: Path to audio file
- **Returns**: Transcribed text or empty string

#### `speak(text, voice_path=None, **kwargs)`
Convert text to speech using configured TTS engine.
- **text**: Text to convert
- **voice_path**: Optional voice reference for cloning
- **Returns**: Path to generated audio file or None

#### `analyze_audio(audio)`
Comprehensive audio analysis.
- **audio**: NumPy array of audio data
- **Returns**: Dictionary with analysis results

### Analysis Functions

#### `detect_speech(audio)`
Detect presence of speech in audio.
- **Returns**: Boolean indicating speech presence

#### `analyze_emotion(audio)`
Basic emotion detection from audio characteristics.
- **Returns**: String emotion label (neutral, excited, calm, sad)

#### `recognize_speaker(audio)`
Speaker identification (placeholder).
- **Returns**: Speaker ID or None

#### `detect_language(audio)`
Language detection (placeholder).
- **Returns**: Language code (default: "en")

### Configuration Functions

#### `set_tts_engine(engine)`
Switch TTS engine.
- **engine**: "chatterbox" or "eleven_labs"
- **Returns**: Boolean success status

#### `cleanup()`
Clean up audio resources.

## Audio Settings

The speech processor uses the following default audio settings:

```python
sample_rate = 16000      # 16 kHz sample rate
chunk_size = 1024        # Audio chunk size
channels = 1             # Mono audio
vad_threshold = 0.01     # Voice activity detection threshold
silence_duration = 2.0   # Seconds of silence to stop recording
min_audio_length = 1.0   # Minimum recording length
```

## Dependencies

The following packages are required:

```
openai-whisper>=1.0.0    # Speech recognition
pyaudio>=0.2.11          # Audio recording (primary)
sounddevice>=0.4.4       # Audio recording (fallback)
soundfile>=0.10.3        # Audio file I/O
numpy>=1.21.0            # Audio processing
```

## Troubleshooting

### Common Issues

1. **"No audio recording library available"**
   - Install PyAudio: `pip install pyaudio`
   - Or install SoundDevice: `pip install sounddevice soundfile`

2. **"Whisper not available"**
   - Install Whisper: `pip install openai-whisper`

3. **CUDA out of memory**
   - Use smaller Whisper model (tiny/base)
   - Set device to "cpu" in config

4. **Poor transcription quality**
   - Use larger Whisper model (medium/large)
   - Ensure good audio quality (quiet environment)
   - Check microphone settings

### Performance Optimization

1. **For faster transcription**: Use "tiny" or "base" models
2. **For better accuracy**: Use "medium" or "large" models
3. **For GPU acceleration**: Ensure CUDA is available and set device to "cuda"
4. **For memory efficiency**: Use smaller models or CPU processing

## Testing

Run the test script to verify functionality:

```bash
python test_speech_recognition.py
```

This will test:
- Model loading
- Audio recording
- Speech transcription
- TTS functionality
- Engine switching
- Basic audio analysis

## Integration with Jarvis

The enhanced speech module integrates seamlessly with the existing Jarvis architecture:

1. **Brain Module**: Can process speech commands and generate responses
2. **NLP Module**: Can analyze transcribed text for intent recognition
3. **Vision Module**: Can combine with visual input for multimodal interaction
4. **System Control**: Can execute voice commands for system operations

## Future Enhancements

Planned improvements include:

1. **Real-time streaming recognition**
2. **Advanced speaker recognition**
3. **Emotion recognition with deep learning**
4. **Multi-language support**
5. **Noise reduction and audio enhancement**
6. **Wake word detection**
7. **Conversation context awareness**

## License

This enhancement maintains compatibility with the existing Jarvis license and includes components from:
- OpenAI Whisper (MIT License)
- PyAudio (MIT License)
- SoundDevice (MIT License)
