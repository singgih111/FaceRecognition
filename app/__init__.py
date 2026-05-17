from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS

from .config import Config
from .models.database import db
from .models.face_recognition import FaceRecognitionSystem

# Initialize face recognition system
face_recognition_system = FaceRecognitionSystem()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    Migrate(app, db)
    CORS(app)
    
    # Register blueprints
    from .routes.main import main_bp
    from .routes.admin import admin_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(admin_bp, url_prefix='/admin')
    
    return app