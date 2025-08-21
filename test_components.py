"""
Quick test script to verify the application components work
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    try:
        from emotion_detector import EmotionDetector
        from face_analyzer import FaceAnalyzer
        print("✅ Emotion detector and face analyzer imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_emotion_detection():
    """Test emotion detection with dummy data"""
    try:
        from emotion_detector import EmotionDetector
        import numpy as np
        
        detector = EmotionDetector()
        
        # Create a dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emotions = detector.detect_emotions(dummy_frame)
        
        if emotions and isinstance(emotions, dict):
            print("✅ Emotion detection working")
            print(f"Sample emotions: {list(emotions.keys())}")
            return True
        else:
            print("❌ Emotion detection failed")
            return False
    except Exception as e:
        print(f"❌ Emotion detection error: {e}")
        return False

def test_face_analysis():
    """Test face analysis with dummy data"""
    try:
        from face_analyzer import FaceAnalyzer
        import numpy as np
        
        analyzer = FaceAnalyzer()
        
        # Create a dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = analyzer.analyze_face(dummy_frame)
        
        print("✅ Face analysis working")
        print(f"Landmarks generated: {len(landmarks) if landmarks else 0}")
        return True
    except Exception as e:
        print(f"❌ Face analysis error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running component tests...")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Emotion Detection", test_emotion_detection),
        ("Face Analysis", test_face_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"⚠️  {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to run.")
        print("Next step: Run 'python app.py' to start the application.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    main()
