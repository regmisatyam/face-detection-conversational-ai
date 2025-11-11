# Facial Expression Recognition and Response

A real-time AI system that detects facial emotions from live video feed and engages in emotionally-aware conversations using Google's Gemini AI.

## Features

**Real-time Emotion Detection**
- Live webcam feed processing
- Face landmark detection with 468 points
- 7 emotion categories: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- Green face mask overlay showing emotion levels

**AI-Powered Conversations**
- Context-aware responses based on detected emotions
- Uses Google Gemini (gemma-2-27b-it) for natural language processing
- Emotion-adaptive conversation style

**Analytics Dashboard**
- Real-time emotion distribution charts
- Emotion timeline tracking
- Face landmark visualization
- Web-based interface

## Installation

1. **Clone the repository**
```bash
git clone [https://github.com/regmisatyam/face-detection-conversational-ai]

cd fdtn
```

2. **Set up Python environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API Keys**
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_api_key
```

To get a Gemini API key:
- Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
- Create a new API key
- Copy it to your `.env` file

## Usage

1. **Start the application**
```bash
python app.py
```

2. **Open your browser**
Navigate to `http://localhost:5500`

3. **Grant camera permissions**
Click "Start Camera" and allow webcam access when prompted

4. **Interact with the AI**
- Your emotions will be detected automatically
- Type messages in the chat to converse with the AI
- The AI will respond based on your current emotional state

## Project Structure

```
fdtn/
├── app.py                 # Main Flask application
├── emotion_detector.py    # Emotion detection module
├── face_analyzer.py       # Face landmark analysis
├── templates/
│   └── index.html        # Web interface
├── requirements.txt      # Python dependencies
├── .env                  # API keys (create this)
└── README.md            # This file
```

## Technology Stack

- **Backend**: Python, Flask, Flask-SocketIO
- **Computer Vision**: OpenCV, MediaPipe, TensorFlow
- **AI/LLM**: Google Gemini (gemma-2-27b-it)
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Real-time Communication**: WebSockets

## Emotion Detection Details

The system uses MediaPipe for face detection and landmark extraction:
- **468 facial landmarks** for precise face mapping
- **7 emotion categories** with confidence scores
- **Real-time processing** at ~5 FPS
- **Green overlay markers** showing emotion intensity across face regions

## AI Conversation Features

- **Emotion-aware responses**: AI adapts its tone based on detected emotions
- **Context preservation**: Maintains conversation history
- **Real-time interaction**: Immediate responses to user input
- **Empathetic AI**: Responds appropriately to emotional states

## Customization

### Adding New Emotions
1. Update `emotion_labels` in `emotion_detector.py`
2. Add corresponding colors in the frontend JavaScript
3. Retrain the emotion model if using custom data

### Modifying AI Responses
Edit the emotion context prompts in `app.py` to change how the AI responds to different emotions.

### Styling Changes
Modify the CSS in `templates/index.html` to customize the interface appearance.

## Troubleshooting

**Camera not working:**
- Ensure browser has camera permissions
- Check if another application is using the camera
- Try refreshing the page

**Emotion detection not accurate:**
- Ensure good lighting conditions
- Position face clearly in camera view
- The current model uses simulated emotions for demo purposes

**AI not responding:**
- Check your Gemini API key in `.env`
- Verify internet connection
- Check browser console for errors

## Development

To extend the system:

1. **Add new emotion models**: Replace the simulated emotions in `emotion_detector.py` with real trained models
2. **Enhance face analysis**: Add more facial feature extraction in `face_analyzer.py`
3. **Improve AI responses**: Fine-tune the prompts and add more context to AI interactions

## License

This project is for educational and research purposes. Please ensure you have proper permissions for any commercial use of the emotion detection technology and AI models.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- Google MediaPipe for face detection
- Google Gemini for AI conversations
- OpenCV community for computer vision tools
- Flask and SocketIO for real-time web framework
