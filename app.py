"""
AI Emotion Detection and Conversation System
Main Flask application that handles real-time emotion detection and AI conversations
"""

import cv2
import numpy as np
import base64
import io
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import google.generativeai as genai
import os
from dotenv import load_dotenv
from emotion_detector import EmotionDetector
from face_analyzer import FaceAnalyzer
import json
from datetime import datetime
import threading
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize components
emotion_detector = EmotionDetector()
face_analyzer = FaceAnalyzer()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemma-2-27b-it')

# Store conversation history and emotion data
conversation_history = []
emotion_data = []

class EmotionConversationAI:
    def __init__(self):
        self.current_emotion = "neutral"
        self.emotion_intensity = 0.5
        
    def generate_response(self, user_message, emotion, intensity):
        """Generate contextual response based on user emotion and message"""
        try:
            # Create emotion-aware prompt
            emotion_context = f"""
            You are an empathetic AI assistant. The user's current emotion is {emotion} with intensity {intensity:.2f}.
            
            Guidelines for responding:
            - If emotion is 'happy' or 'joy': Be enthusiastic and positive
            - If emotion is 'sad' or 'angry': Be supportive and understanding
            - If emotion is 'fear' or 'surprise': Be reassuring and calm
            - If emotion is 'disgust': Be neutral and redirect positively
            - If emotion is 'neutral': Be balanced and engaging
            
            User message: {user_message}
            
            Respond appropriately to their emotion and message. Keep responses concise but meaningful.
            """
            
            response = model.generate_content(emotion_context)
            return response.text
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I understand your feelings. How can I help you today?"

# Initialize AI conversation system
ai_conversation = EmotionConversationAI()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with emotion context"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        current_emotion = data.get('emotion', 'neutral')
        emotion_intensity = data.get('intensity', 0.5)
        
        # Generate AI response
        ai_response = ai_conversation.generate_response(
            user_message, current_emotion, emotion_intensity
        )
        
        # Store conversation
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'ai_response': ai_response,
            'emotion': current_emotion,
            'intensity': emotion_intensity
        }
        conversation_history.append(conversation_entry)
        
        return jsonify({
            'response': ai_response,
            'emotion': current_emotion,
            'intensity': emotion_intensity
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'error': 'Failed to process message'}), 500

@socketio.on('video_frame')
def handle_video_frame(data):
    """Process video frames for emotion detection"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect emotions and analyze face
        emotions = emotion_detector.detect_emotions(frame)
        face_landmarks = face_analyzer.analyze_face(frame)
        
        # Get dominant emotion
        if emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name = dominant_emotion[0]
            emotion_confidence = float(dominant_emotion[1])  # Convert to Python float
            
            # Convert emotions to JSON-serializable format
            serializable_emotions = {k: float(v) for k, v in emotions.items()}
            
            # Convert face landmarks to JSON-serializable format
            serializable_landmarks = []
            if face_landmarks:
                for face_data in face_landmarks:
                    serializable_face = {}
                    if 'landmarks' in face_data:
                        serializable_face['landmarks'] = [
                            {'x': float(landmark['x']), 'y': float(landmark['y'])} 
                            for landmark in face_data['landmarks']
                        ]
                    if 'face_rect' in face_data:
                        x, y, w, h = face_data['face_rect']
                        serializable_face['face_rect'] = [int(x), int(y), int(w), int(h)]
                    if 'emotion_features' in face_data:
                        serializable_face['emotion_features'] = {
                            k: float(v) for k, v in face_data['emotion_features'].items()
                        }
                    serializable_landmarks.append(serializable_face)
            
            # Store emotion data
            emotion_entry = {
                'timestamp': datetime.now().isoformat(),
                'emotion': emotion_name,
                'confidence': emotion_confidence,
                'all_emotions': serializable_emotions
            }
            emotion_data.append(emotion_entry)
            
            # Emit results to frontend
            emit('emotion_detected', {
                'emotion': emotion_name,
                'confidence': emotion_confidence,
                'all_emotions': serializable_emotions,
                'face_landmarks': serializable_landmarks
            })
            
    except Exception as e:
        print(f"Video frame processing error: {e}")
        emit('error', {'message': 'Failed to process video frame'})

@app.route('/api/emotion_history')
def get_emotion_history():
    """Get emotion detection history"""
    return jsonify(emotion_data[-50:])  # Return last 50 entries

@app.route('/api/conversation_history')
def get_conversation_history():
    """Get conversation history"""
    return jsonify(conversation_history[-20:])  # Return last 20 entries

if __name__ == '__main__':
    print("Starting AI Emotion Detection System...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5500)
