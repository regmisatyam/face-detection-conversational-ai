"""
Demo script to test emotion detection functionality
"""

import cv2
import numpy as np
from emotion_detector import EmotionDetector
from face_analyzer import FaceAnalyzer
import time

def main():
    # Initialize components
    emotion_detector = EmotionDetector()
    face_analyzer = FaceAnalyzer()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting emotion detection demo...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect emotions
        emotions = emotion_detector.detect_emotions(frame)
        
        # Analyze face
        landmarks_data = face_analyzer.analyze_face(frame)
        
        # Create emotion overlay
        if emotions and landmarks_data:
            overlay_frame = face_analyzer.get_face_emotions_overlay(frame, landmarks_data, emotions)
        else:
            overlay_frame = frame
        
        # Draw face mesh
        if landmarks_data:
            overlay_frame = face_analyzer.draw_face_mesh(overlay_frame, landmarks_data)
        
        # Display emotion information
        if emotions:
            y = 30
            for emotion, confidence in emotions.items():
                text = f"{emotion}: {confidence:.2f}"
                cv2.putText(overlay_frame, text, (10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y += 30
        
        # Show frame
        cv2.imshow('Emotion Detection Demo', overlay_frame)
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
