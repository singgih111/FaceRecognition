from flask import Blueprint, render_template, jsonify, request
import cv2
import numpy as np
import base64
from .. import face_recognition_system
from ..models.database import db, User, FaceEncoding

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Render the main page with video stream"""
    return render_template('index.html')

@main_bp.route('/api/recognize', methods=['POST'])
def recognize_face():
    """
    Recognize faces in the uploaded image
    
    Expects a JSON with base64 encoded image
    Returns list of recognized faces with their locations
    """
    try:
        # Get image data from request
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Convert base64 to numpy array
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Detect faces
        faces, boxes = face_recognition_system.detect_faces(image)
        
        if not faces:
            return jsonify({'faces': []})
        
        results = []
        # Get all users and their face encodings
        users = User.query.all()
        
        for face, box in zip(faces, boxes):
            # Get embedding for detected face
            embedding = face_recognition_system.get_face_embedding(face)
            if embedding is None:
                continue
            
            # Compare with stored faces
            best_match = None
            best_score = 0.0  # Convert to Python float
            
            for user in users:
                for stored_encoding in user.face_encodings:
                    stored_embedding = stored_encoding.get_encoding()
                    score = float(face_recognition_system.compare_faces(embedding, stored_embedding))  # Convert to Python float
                    
                    if score > best_score:
                        best_score = score
                        best_match = user
            
            # Convert numpy int values to Python int
            box = [int(x) for x in box]
            
            # Add result
            result = {
                'box': box,
                'recognized': bool(best_score > 0.6),  # Convert to Python bool
                'name': best_match.name if best_match and best_score > 0.6 else 'Unknown',
                'confidence': best_score
            }
            results.append(result)
        
        return jsonify({'faces': results})
    
    except Exception as e:
        print(f"Error in recognize_face: {str(e)}")
        return jsonify({'error': str(e)}), 500 