import cv2
import numpy as np
import time
from modules.camera import camera
from modules.ai_vision import ai_vision_system
from modules.visualization import vision_visualizer

def test_camera():
    """Test camera initialization and frame capture"""
    print("\nTesting camera module...")
    
    try:
        # Start camera
        if not camera.start():
            print("❌ Failed to start camera")
            return False
            
        # Get a test frame
        frame = camera.get_frame()
        if frame is None:
            print("❌ Failed to capture frame")
            return False
            
        print(f"✓ Camera working (Resolution: {frame.shape[1]}x{frame.shape[0]})")
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {str(e)}")
        return False
    finally:
        camera.stop()

def test_ai_vision():
    """Test AI vision system components"""
    print("\nTesting AI Vision System...")
    
    # Start camera for testing
    if not camera.start():
        print("❌ Cannot test AI Vision - Camera failed to start")
        return False
        
    try:
        # Initialize AI Vision
        ai_vision_system.start_processing()
        
        # Get a test frame
        frame = camera.get_frame()
        if frame is None:
            print("❌ Cannot test AI Vision - No frame available")
            return False
            
        # Process frame
        results = ai_vision_system.process_frame(frame)
        if results is None:
            print("❌ Failed to process frame")
            return False
            
        # Check results structure
        expected_keys = ['timestamp', 'objects', 'expressions', 'gestures', 'scene_3d']
        missing_keys = [key for key in expected_keys if key not in results]
        
        if missing_keys:
            print(f"❌ Missing expected results keys: {missing_keys}")
            return False
            
        print("✓ AI Vision System working")
        print(f"  - Objects detected: {len(results['objects'])}")
        print(f"  - Faces analyzed: {len(results['expressions'])}")
        print(f"  - Gestures recognized: {len(results['gestures'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Vision test failed: {str(e)}")
        return False
    finally:
        ai_vision_system.stop_processing()
        camera.stop()

def test_visualization():
    """Test visualization system"""
    print("\nTesting visualization system...")
    
    try:
        # Create test frame and results
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_results = {
            'objects': [{
                'class': 'test_object',
                'confidence': 0.95,
                'bbox': [100, 100, 200, 200]
            }],
            'expressions': [{
                'box': [300, 100, 400, 200],
                'emotions': {'happy': 0.8, 'neutral': 0.2}
            }],
            'gestures': [{
                'landmarks': [{'x': 0.7, 'y': 0.7, 'z': 0} for _ in range(21)],
                'gesture': 'peace'
            }]
        }
        
        # Test drawing results
        result_frame = vision_visualizer.draw_detection_results(test_frame, test_results)
        if result_frame is None:
            print("❌ Failed to draw detection results")
            return False
            
        # Test metrics overlay
        test_metrics = {'fps': 30.0, 'avg_processing_time': 0.033}
        final_frame = vision_visualizer.add_performance_metrics(result_frame, test_metrics)
        if final_frame is None:
            print("❌ Failed to add performance metrics")
            return False
            
        print("✓ Visualization system working")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {str(e)}")
        return False

def test_full_pipeline():
    """Test the complete vision pipeline"""
    print("\nTesting full vision pipeline...")
    
    try:
        # Start camera
        if not camera.start():
            print("❌ Full pipeline test failed - Camera not available")
            return False
            
        # Start AI Vision
        ai_vision_system.start_processing()
        
        print("Running pipeline test for 5 seconds...")
        start_time = time.time()
        frames_processed = 0
        
        while time.time() - start_time < 5:
            # Get frame
            frame = camera.get_frame()
            if frame is None:
                continue
                
            # Process frame
            results = ai_vision_system.process_frame(frame)
            if results is None:
                continue
                
            # Visualize results
            frame = vision_visualizer.draw_detection_results(frame, results)
            
            # Add metrics
            metrics = ai_vision_system.get_performance_metrics()
            frame = vision_visualizer.add_performance_metrics(frame, metrics)
            
            # Display frame
            cv2.imshow('Pipeline Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frames_processed += 1
            
        cv2.destroyAllWindows()
        
        fps = frames_processed / 5
        print(f"✓ Full pipeline working at {fps:.1f} FPS")
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {str(e)}")
        return False
    finally:
        ai_vision_system.stop_processing()
        camera.stop()

def main():
    """Run all tests"""
    print("Starting AI Vision System Tests...")
    
    # Run individual component tests
    camera_ok = test_camera()
    vision_ok = test_ai_vision()
    viz_ok = test_visualization()
    
    # Only run full pipeline if component tests pass
    if camera_ok and vision_ok and viz_ok:
        pipeline_ok = test_full_pipeline()
    else:
        pipeline_ok = False
        print("\n❌ Skipping full pipeline test due to component test failures")
    
    # Print summary
    print("\nTest Summary:")
    print(f"Camera Module: {'✓' if camera_ok else '❌'}")
    print(f"AI Vision System: {'✓' if vision_ok else '❌'}")
    print(f"Visualization: {'✓' if viz_ok else '❌'}")
    print(f"Full Pipeline: {'✓' if pipeline_ok else '❌'}")
    
    if all([camera_ok, vision_ok, viz_ok, pipeline_ok]):
        print("\n✓ All tests passed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
