import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/face_recognition')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Face Recognition
    FACE_RECOGNITION_THRESHOLD = 0.6  # Threshold for face matching confidence
    FACE_DETECTION_CONFIDENCE = 0.9   # Threshold for face detection confidence
    
    # File Upload
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Ensure upload directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True) 