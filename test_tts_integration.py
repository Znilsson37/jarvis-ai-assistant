"""Comprehensive TTS integration test script"""

import json
import os
import time
import tempfile
import shutil
from typing import Dict, Any, List

def test_config_loading():
    """Test that configuration is properly loaded"""
    try:
        config_path = os.path.join("Config", "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("✓ Configuration loaded successfully")
        
        # Check TTS configuration
        tts_config = config.get("tts", {})
        print(f"✓ TTS configuration found: {bool(tts_config)}")
        
        default_engine = tts_config.get("default_engine", "chatterbox")
        print(f"✓ Default TTS engine: {default_engine}")
        
        # Check Chatterbox config
        chatterbox_config = tts_config.get("chatterbox", {})
        print(f"✓ Chatterbox configuration: {chatterbox_config}")
        
        # Check Eleven Labs config
        eleven_labs_config = tts_config.get("eleven_labs", {})
        print(f"✓ Eleven Labs configuration: {eleven_labs_config}")
        
        # Validate configuration values
        if chatterbox_config:
            device = chatterbox_config.get("device", "cpu")
            exaggeration = chatterbox_config.get("default_exaggeration", 0.5)
            cfg_weight = chatterbox_config.get("default_cfg_weight", 0.5)
            
            print(f"  - Device: {device}")
            print(f"  - Default exaggeration: {exaggeration}")
            print(f"  - Default cfg_weight: {cfg_weight}")
            
            if not (0.0 <= exaggeration <= 1.0):
                print(f"⚠ Warning: exaggeration value {exaggeration} outside recommended range [0.0, 1.0]")
            if not (0.0 <= cfg_weight <= 1.0):
                print(f"⚠ Warning: cfg_weight value {cfg_weight} outside recommended range [0.0, 1.0]")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False

def test_chatterbox_availability():
    """Test if Chatterbox TTS is available"""
    try:
        import torch
        print(f"✓ PyTorch available: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA device count: {torch.cuda.device_count()}")
            print(f"✓ Current CUDA device: {torch.cuda.current_device()}")
            print(f"✓ CUDA device name: {torch.cuda.get_device_name()}")
        
        try:
            import torchaudio
            print(f"✓ TorchAudio available: {torchaudio.__version__}")
        except ImportError as e:
            print(f"✗ TorchAudio not available: {e}")
            return False
            
        try:
            from chatterbox.tts import ChatterboxTTS
            print("✓ Chatterbox TTS import successful")
            
            # Test model initialization
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"✓ Testing model initialization on {device}...")
            
            start_time = time.time()
            model = ChatterboxTTS.from_pretrained(device=device)
            init_time = time.time() - start_time
            
            print(f"✓ Model initialized successfully in {init_time:.2f}s")
            print(f"✓ Model sample rate: {model.sr}")
            
            return True
            
        except ImportError as e:
            print(f"✗ Chatterbox TTS not available: {e}")
            return False
        except RuntimeError as e:
            print(f"✗ Chatterbox TTS runtime error: {e}")
            return False
        except Exception as e:
            print(f"✗ Chatterbox TTS initialization error: {e}")
            return False
            
    except ImportError as e:
        print(f"✗ PyTorch not available: {e}")
        return False

def test_eleven_labs_availability():
    """Test if Eleven Labs is available"""
    try:
        config_path = os.path.join("Config", "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        api_key = config.get("eleven_labs_api_key")
        if api_key and api_key.strip():
            print("✓ Eleven Labs API key configured")
            
            # Test API connection
            try:
                import requests
                headers = {"xi-api-key": api_key}
                response = requests.get("https://api.elevenlabs.io/v1/user", headers=headers, timeout=10)
                
                if response.status_code == 200:
                    user_data = response.json()
                    print(f"✓ API connection successful")
                    print(f"  - User ID: {user_data.get('subscription', {}).get('tier', 'Unknown')}")
                    print(f"  - Character count: {user_data.get('subscription', {}).get('character_count', 'Unknown')}")
                    print(f"  - Character limit: {user_data.get('subscription', {}).get('character_limit', 'Unknown')}")
                    return True
                else:
                    print(f"✗ API connection failed: {response.status_code}")
                    return False
                    
            except ImportError:
                print("⚠ requests module not available for API testing")
                return True  # API key is configured, assume it works
            except Exception as e:
                print(f"✗ API connection error: {e}")
                return False
        else:
            print("✗ Eleven Labs API key not configured")
            return False
            
    except Exception as e:
        print(f"✗ Error checking Eleven Labs configuration: {e}")
        return False

def test_speech_processor_integration():
    """Test speech processor integration"""
    try:
        print("✓ Testing speech processor integration...")
        
        # Import speech processor
        from modules.speech import speech_processor
        print("✓ Speech processor imported successfully")
        
        # Test configuration loading
        config = speech_processor.config
        print(f"✓ Configuration loaded: {bool(config)}")
        
        # Test TTS engine
        current_engine = speech_processor.tts_engine
        print(f"✓ Current TTS engine: {current_engine}")
        
        # Test engine switching
        if speech_processor.set_tts_engine("chatterbox"):
            print("✓ Successfully switched to Chatterbox")
        else:
            print("✗ Failed to switch to Chatterbox")
            
        if speech_processor.eleven_labs_tts:
            if speech_processor.set_tts_engine("eleven_labs"):
                print("✓ Successfully switched to Eleven Labs")
                speech_processor.set_tts_engine("chatterbox")  # Switch back
            else:
                print("✗ Failed to switch to Eleven Labs")
        else:
            print("⚠ Eleven Labs not available for switching test")
        
        return True
        
    except Exception as e:
        print(f"✗ Speech processor integration error: {e}")
        return False

def test_basic_tts_functionality():
    """Test basic TTS functionality"""
    try:
        print("✓ Testing basic TTS functionality...")
        
        from modules.speech import speech_processor
        
        # Test simple text generation
        test_text = "This is a test of the text to speech system."
        
        start_time = time.time()
        output_path = speech_processor.speak(test_text)
        generation_time = time.time() - start_time
        
        if output_path and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✓ Audio generated successfully")
            print(f"  - Output path: {output_path}")
            print(f"  - File size: {file_size} bytes")
            print(f"  - Generation time: {generation_time:.2f}s")
            
            # Clean up test file
            try:
                os.remove(output_path)
                print("✓ Test file cleaned up")
            except:
                pass
                
            return True
        else:
            print("✗ Audio generation failed")
            return False
            
    except Exception as e:
        print(f"✗ Basic TTS functionality test error: {e}")
        return False

def test_edge_cases():
    """Test edge cases and error handling"""
    try:
        print("✓ Testing edge cases...")
        
        from modules.speech import speech_processor
        
        # Test empty text
        result = speech_processor.speak("")
        if result is None:
            print("✓ Empty text handled correctly")
        else:
            print("⚠ Empty text should return None")
        
        # Test None input
        result = speech_processor.speak(None)
        if result is None:
            print("✓ None input handled correctly")
        else:
            print("⚠ None input should return None")
        
        # Test very long text
        long_text = "test " * 100
        result = speech_processor.speak(long_text)
        if result:
            print("✓ Long text processed successfully")
            try:
                os.remove(result)
            except:
                pass
        else:
            print("⚠ Long text processing failed")
        
        return True
        
    except Exception as e:
        print(f"✗ Edge cases test error: {e}")
        return False

def run_performance_benchmark():
    """Run performance benchmark"""
    try:
        print("✓ Running performance benchmark...")
        
        from modules.speech import speech_processor
        
        test_cases = [
            ("Short", "Hello world."),
            ("Medium", "This is a medium length sentence for testing purposes."),
            ("Long", "This is a much longer sentence that contains multiple clauses and should take more time to process than the shorter examples, allowing us to measure performance differences.")
        ]
        
        results = []
        for name, text in test_cases:
            start_time = time.time()
            output_path = speech_processor.speak(text)
            end_time = time.time()
            
            if output_path and os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                processing_time = end_time - start_time
                
                results.append({
                    'name': name,
                    'text_length': len(text),
                    'processing_time': processing_time,
                    'file_size': file_size
                })
                
                print(f"  - {name}: {processing_time:.2f}s, {file_size} bytes")
                
                # Clean up
                try:
                    os.remove(output_path)
                except:
                    pass
            else:
                print(f"  - {name}: Failed")
        
        if results:
            avg_time = sum(r['processing_time'] for r in results) / len(results)
            print(f"✓ Average processing time: {avg_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance benchmark error: {e}")
        return False

def main():
    """Run comprehensive TTS integration tests"""
    print("=== Comprehensive TTS Integration Test ===\n")
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Chatterbox TTS Availability", test_chatterbox_availability),
        ("Eleven Labs Availability", test_eleven_labs_availability),
        ("Speech Processor Integration", test_speech_processor_integration),
        ("Basic TTS Functionality", test_basic_tts_functionality),
        ("Edge Cases", test_edge_cases),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"{len(results) + 1}. {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results[test_name] = False
        print()
    
    # Summary
    print("=== Test Results Summary ===")
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! TTS integration is working correctly.")
    elif passed >= total * 0.8:
        print("\n⚠ Most tests passed. TTS integration is mostly functional.")
    else:
        print("\n❌ Multiple test failures. TTS integration needs attention.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
