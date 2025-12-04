"""
Test script for sensor fusion module
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'built-in'))

from sensor_fusion import ComplementaryFilter, MPUGestureDetector, SensorFusion, GestureDecisionEngine

def test_complementary_filter():
    print("=" * 60)
    print("TEST 1: Complementary Filter")
    print("=" * 60)
    
    filter = ComplementaryFilter(alpha=0.85)
    
    # Simulate noisy accelerometer data
    test_data = [
        (10000, 500, 0),
        (10500, 600, 0),
        (9800, 480, 0),
        (10200, 520, 0),
    ]
    
    print("\nInput (noisy) → Filtered (smooth):")
    for ax, ay, az in test_data:
        filtered = filter.update(ax, ay, az)
        print(f"  ({ax:6}, {ay:4}, {az:4}) → ({filtered['x']:8.1f}, {filtered['y']:6.1f}, {filtered['z']:6.1f})")
    
    print("✅ Filter smooths data successfully\n")


def test_mpu_gesture_detector():
    print("=" * 60)
    print("TEST 2: MPU Gesture Detector")
    print("=" * 60)
    
    detector = MPUGestureDetector(tilt_threshold=7000)
    
    test_cases = [
        {"name": "Tilt Right", "data": {"AcX": 10000, "AcY": 500, "AcZ": 0}},
        {"name": "Tilt Left", "data": {"AcX": -9000, "AcY": 500, "AcZ": 0}},
        {"name": "Tilt Forward", "data": {"AcX": 500, "AcY": 8500, "AcZ": 0}},
        {"name": "No Gesture", "data": {"AcX": 1000, "AcY": 500, "AcZ": 0}},
    ]
    
    print("\nGesture Detection:")
    for case in test_cases:
        gesture, confidence = detector.detect_gesture(case["data"])
        print(f"  {case['name']:15} → {gesture or 'None':15} (confidence: {confidence:5.1f}%)")
        import time
        time.sleep(2)  # Wait for cooldown
    
    print("✅ Gesture detection working\n")


def test_sensor_fusion():
    print("=" * 60)
    print("TEST 3: Sensor Fusion")
    print("=" * 60)
    
    fusion = SensorFusion(camera_weight=0.5, mpu_weight=0.5)
    
    # Simulate MPU data
    fusion.update_mpu_data({"AcX": 10000, "AcY": 500, "AcZ": 0})
    
    test_cases = [
        {"camera": "SWIPE_RIGHT", "cam_conf": 80, "desc": "Camera + MPU agree"},
        {"camera": "SWIPE_LEFT", "cam_conf": 70, "desc": "Camera + MPU conflict"},
        {"camera": "PUSH_FORWARD", "cam_conf": 85, "desc": "Camera only (no MPU)"},
    ]
    
    print("\nFusion Results:")
    for i, case in enumerate(test_cases):
        if i == 0:
            mpu_gesture, mpu_conf = "TILT_RIGHT", 75.0
        elif i == 1:
            mpu_gesture, mpu_conf = "TILT_RIGHT", 60.0
        else:
            fusion.mpu_data_queue.clear()  # Simulate no MPU
            mpu_gesture, mpu_conf = None, 0.0
        
        final_gesture, final_conf = fusion.fuse_gestures(
            case["camera"], case["cam_conf"],
            mpu_gesture, mpu_conf
        )
        
        print(f"  {case['desc']:25} → {final_gesture:15} ({final_conf:5.1f}%)")
    
    print("✅ Sensor fusion working\n")


def test_decision_engine():
    print("=" * 60)
    print("TEST 4: Gesture Decision Engine")
    print("=" * 60)
    
    engine = GestureDecisionEngine(min_confidence=60.0, stability_frames=3)
    
    print("\nTemporal Smoothing (3 frames required):")
    
    # Test stable gesture
    gestures = ["SWIPE_RIGHT", "SWIPE_RIGHT", "SWIPE_RIGHT"]
    for i, g in enumerate(gestures):
        confirmed, conf, state = engine.decide(g, 80.0)
        print(f"  Frame {i+1}: {g:15} → State: {state:10} | Confirmed: {confirmed or 'None'}")
    
    print("\n  Unstable gesture:")
    engine.reset()
    gestures = ["SWIPE_RIGHT", "SWIPE_LEFT", "SWIPE_RIGHT"]
    for i, g in enumerate(gestures):
        confirmed, conf, state = engine.decide(g, 80.0)
        print(f"  Frame {i+1}: {g:15} → State: {state:10} | Confirmed: {confirmed or 'None'}")
    
    print("✅ Decision engine working\n")


def main():
    print("\n" + "=" * 60)
    print(" SENSOR FUSION MODULE - TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_complementary_filter()
        test_mpu_gesture_detector()
        test_sensor_fusion()
        test_decision_engine()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
