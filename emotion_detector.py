"""
Emotion Detection Module
Uses OpenCV for face detection and simulated emotion recognition
"""

import cv2
import numpy as np
import random
import time

class EmotionDetector:
    def __init__(self):
        # Load OpenCV's pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Emotion labels
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
    def detect_emotions(self, frame):
        """Detect emotions in a frame using face detection and simulation"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            emotions = {}
            
            if len(faces) > 0:
                # Use the first detected face
                (x, y, w, h) = faces[0]
                
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                if face_roi.size > 0:
                    # Simulate realistic emotion detection
                    emotions = self._simulate_realistic_emotions(face_roi)
                else:
                    emotions = self._get_default_emotions()
            else:
                emotions = self._get_default_emotions()
            
            return emotions
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return self._get_default_emotions()
    
    
    def _simulate_realistic_emotions(self, face_roi):
        """Simulate realistic emotion detection based on basic face analysis"""
        # Analyze basic face features for emotion simulation
        
        # Calculate some basic features
        height, width = face_roi.shape
        
        # Simple heuristics for emotion simulation
        # In a real implementation, you'd use ML models trained on emotion datasets
        
        # Base emotions with realistic variations
        base_emotions = {
            'happy': 0.2 + random.random() * 0.4,
            'neutral': 0.3 + random.random() * 0.3,
            'surprise': 0.1 + random.random() * 0.2,
            'sad': 0.05 + random.random() * 0.15,
            'angry': 0.05 + random.random() * 0.1,
            'fear': 0.02 + random.random() * 0.08,
            'disgust': 0.02 + random.random() * 0.05
        }
        
        # Add some time-based variation for demo realism
        time_factor = time.time() % 10  # 10-second cycle
        if time_factor < 2:
            base_emotions['happy'] += 0.2
        elif time_factor < 4:
            base_emotions['neutral'] += 0.2
        elif time_factor < 6:
            base_emotions['surprise'] += 0.1
        else:
            base_emotions['sad'] += 0.1
        
        # Normalize to sum to 1
        total = sum(base_emotions.values())
        normalized_emotions = {k: v/total for k, v in base_emotions.items()}
        
        return normalized_emotions
    
    def _get_default_emotions(self):
        """Return default emotion values when detection fails"""
        return {
            'neutral': 0.7,
            'happy': 0.1,
            'sad': 0.05,
            'angry': 0.05,
            'surprise': 0.05,
            'fear': 0.03,
            'disgust': 0.02
        }
    
    def get_dominant_emotion(self, emotions):
        """Get the dominant emotion from emotion scores"""
        if not emotions:
            return 'neutral', 0.0
        
        dominant = max(emotions.items(), key=lambda x: x[1])
        return dominant[0], dominant[1]
