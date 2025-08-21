"""
Face Analysis Module
Uses OpenCV for basic face detection and landmark simulation
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple

class FaceAnalyzer:
    def __init__(self):
        # Load OpenCV's pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Simulated face regions for visualization
        self.face_regions = {
            'left_eye': [],
            'right_eye': [],
            'mouth': [],
            'eyebrows': [],
            'nose': []
        }
    
    def analyze_face(self, frame):
        """Analyze face and return simulated landmark data"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            landmarks_data = []
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Generate simulated landmarks for the face
                    landmarks = self._generate_simulated_landmarks(x, y, w, h)
                    
                    landmarks_data.append({
                        'landmarks': landmarks,
                        'regions': self._get_region_coordinates(landmarks),
                        'emotion_features': self._extract_emotion_features(landmarks),
                        'face_rect': (x, y, w, h)
                    })
            
            return landmarks_data
            
        except Exception as e:
            print(f"Error in face analysis: {e}")
            return []
    
    def _generate_simulated_landmarks(self, x, y, w, h):
        """Generate simulated facial landmarks for a detected face"""
        landmarks = []
        
        # Generate landmarks in a grid pattern across the face
        # This simulates the 468 MediaPipe landmarks
        
        # Face outline (jawline)
        for i in range(17):
            px = x + (i / 16.0) * w
            py = y + h * 0.8 + np.sin(i * 0.3) * h * 0.1
            landmarks.append({'x': px, 'y': py})
        
        # Left eyebrow
        for i in range(5):
            px = x + w * (0.2 + i * 0.1)
            py = y + h * 0.25
            landmarks.append({'x': px, 'y': py})
        
        # Right eyebrow
        for i in range(5):
            px = x + w * (0.6 + i * 0.1)
            py = y + h * 0.25
            landmarks.append({'x': px, 'y': py})
        
        # Left eye
        for i in range(6):
            angle = i * np.pi / 3
            px = x + w * 0.3 + np.cos(angle) * w * 0.05
            py = y + h * 0.35 + np.sin(angle) * h * 0.03
            landmarks.append({'x': px, 'y': py})
        
        # Right eye
        for i in range(6):
            angle = i * np.pi / 3
            px = x + w * 0.7 + np.cos(angle) * w * 0.05
            py = y + h * 0.35 + np.sin(angle) * h * 0.03
            landmarks.append({'x': px, 'y': py})
        
        # Nose
        for i in range(5):
            px = x + w * 0.5
            py = y + h * (0.4 + i * 0.05)
            landmarks.append({'x': px, 'y': py})
        
        # Mouth
        for i in range(8):
            angle = i * np.pi / 4
            px = x + w * 0.5 + np.cos(angle) * w * 0.08
            py = y + h * 0.7 + np.sin(angle) * h * 0.04
            landmarks.append({'x': px, 'y': py})
        
        # Fill the rest with distributed points across the face
        remaining_points = 468 - len(landmarks)
        for i in range(remaining_points):
            px = x + (i % 20) * w / 20 + np.random.normal(0, w * 0.02)
            py = y + (i // 20) * h / (remaining_points // 20 + 1) + np.random.normal(0, h * 0.02)
            landmarks.append({'x': px, 'y': py})
        
        return landmarks[:468]  # Limit to 468 points
    
    def _get_region_coordinates(self, landmarks):
        """Get coordinates for specific face regions based on simulated landmarks"""
        regions = {}
        
        if len(landmarks) >= 50:
            # Map landmark indices to face regions (simplified)
            regions['left_eye'] = landmarks[33:39] if len(landmarks) > 39 else landmarks[10:16]
            regions['right_eye'] = landmarks[39:45] if len(landmarks) > 45 else landmarks[16:22]
            regions['mouth'] = landmarks[45:53] if len(landmarks) > 53 else landmarks[22:30]
            regions['eyebrows'] = landmarks[17:27] if len(landmarks) > 27 else landmarks[5:15]
            regions['nose'] = landmarks[27:32] if len(landmarks) > 32 else landmarks[30:35]
        else:
            # Fallback for insufficient landmarks
            regions = {region: [] for region in self.face_regions.keys()}
        
        return regions
    
    def _extract_emotion_features(self, landmarks):
        """Extract features that indicate emotions from simulated landmarks"""
        try:
            features = {}
            
            if len(landmarks) >= 50:
                # Mouth curve (smile detection) - using simulated mouth landmarks
                mouth_points = landmarks[45:53] if len(landmarks) > 53 else landmarks[22:30]
                if len(mouth_points) >= 4:
                    mouth_left = mouth_points[0]
                    mouth_right = mouth_points[3]
                    mouth_top = mouth_points[1]
                    mouth_bottom = mouth_points[2]
                    
                    mouth_width = abs(mouth_right['x'] - mouth_left['x'])
                    mouth_height = abs(mouth_bottom['y'] - mouth_top['y'])
                    
                    features['mouth_curve'] = mouth_height / mouth_width if mouth_width > 0 else 0
                
                # Eye openness - using simulated eye landmarks
                left_eye_points = landmarks[33:39] if len(landmarks) > 39 else landmarks[10:16]
                right_eye_points = landmarks[39:45] if len(landmarks) > 45 else landmarks[16:22]
                
                if len(left_eye_points) >= 4 and len(right_eye_points) >= 4:
                    left_eye_openness = abs(left_eye_points[1]['y'] - left_eye_points[3]['y'])
                    right_eye_openness = abs(right_eye_points[1]['y'] - right_eye_points[3]['y'])
                    features['eye_openness'] = (left_eye_openness + right_eye_openness) / 2
                
                # Eyebrow position (surprise/anger indicator)
                eyebrow_points = landmarks[17:27] if len(landmarks) > 27 else landmarks[5:15]
                if len(eyebrow_points) >= 4:
                    avg_eyebrow_y = sum(p['y'] for p in eyebrow_points[:4]) / 4
                    avg_eye_y = sum(p['y'] for p in (left_eye_points + right_eye_points)[:8]) / 8
                    features['eyebrow_height'] = abs(avg_eyebrow_y - avg_eye_y)
            
            return features
            
        except Exception as e:
            print(f"Error extracting emotion features: {e}")
            return {}
    
    def draw_face_mesh(self, frame, landmarks_data):
        """Draw simulated face mesh on frame"""
        try:
            if landmarks_data:
                for face_data in landmarks_data:
                    landmarks = face_data['landmarks']
                    
                    # Draw landmarks as small circles
                    for landmark in landmarks:
                        x, y = int(landmark['x']), int(landmark['y'])
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    
                    # Draw face rectangle if available
                    if 'face_rect' in face_data:
                        x, y, w, h = face_data['face_rect']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            return frame
            
        except Exception as e:
            print(f"Error drawing face mesh: {e}")
            return frame
    
    def get_face_emotions_overlay(self, frame, landmarks_data, emotions):
        """Create an overlay showing emotion levels on face regions"""
        overlay = frame.copy()
        
        try:
            if landmarks_data and emotions:
                face_data = landmarks_data[0]
                landmarks = face_data['landmarks']
                
                # Define colors for different emotions
                emotion_colors = {
                    'happy': (0, 255, 0),      # Green
                    'sad': (255, 0, 0),        # Blue
                    'angry': (0, 0, 255),      # Red
                    'surprise': (255, 255, 0), # Yellow
                    'fear': (128, 0, 128),     # Purple
                    'disgust': (0, 128, 128),  # Teal
                    'neutral': (128, 128, 128) # Gray
                }
                
                # Get dominant emotion
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                emotion_name = dominant_emotion[0]
                emotion_intensity = dominant_emotion[1]
                
                # Draw emotion indicators on face landmarks
                if emotion_name in emotion_colors:
                    color = emotion_colors[emotion_name]
                    
                    # Draw circles on all landmarks with intensity-based sizing
                    for landmark in landmarks:
                        x = int(landmark['x'])
                        y = int(landmark['y'])
                        
                        # Draw circles with size based on emotion intensity
                        radius = max(1, int(2 + emotion_intensity * 4))
                        cv2.circle(overlay, (x, y), radius, color, -1)
                
                # Add emotion text overlay
                if 'face_rect' in face_data:
                    x, y, w, h = face_data['face_rect']
                    text_y = y - 10 if y > 30 else y + h + 25
                    cv2.putText(overlay, f"{emotion_name}: {emotion_intensity:.2f}", 
                               (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add emotion bars for all emotions
                bar_x = 10
                bar_y = 50
                for emotion, confidence in emotions.items():
                    if emotion in emotion_colors:
                        # Draw emotion bar
                        bar_width = int(confidence * 200)
                        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), 
                                    emotion_colors[emotion], -1)
                        cv2.putText(overlay, f"{emotion}: {confidence:.2f}", 
                                   (bar_x + 210, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                   (255, 255, 255), 1)
                        bar_y += 25
            
            return overlay
            
        except Exception as e:
            print(f"Error creating emotion overlay: {e}")
            return frame
