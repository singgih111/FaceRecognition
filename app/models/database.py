from datetime import datetime
import numpy as np
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    face_encodings = db.relationship('FaceEncoding', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.name}>'

class FaceEncoding(db.Model):
    __tablename__ = 'face_encodings'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    encoding_vector = db.Column(db.LargeBinary, nullable=False)  # Store face encoding as binary
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_encoding(self, encoding):
        """Convert numpy array to binary for storage"""
        self.encoding_vector = encoding.tobytes()
    
    def get_encoding(self):
        """Convert binary back to numpy array"""
        return np.frombuffer(self.encoding_vector, dtype=np.float32)

    def __repr__(self):
        return f'<FaceEncoding user_id={self.user_id}>' 