import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask
from flask_migrate import Migrate
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

# Add the parent directory to Python path to find the app package
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from app import create_app
from app.models.database import db, User, FaceEncoding

def create_tables(app):
    """Create database tables"""
    with app.app_context():
        # Drop existing tables if they exist
        db.drop_all()
        
        # Create all tables
        db.create_all()
        
        # Verify tables were created
        inspector = db.inspect(db.engine)
        
        # Expected table structure
        expected_tables = {
            'users': [
                'id', 'name', 'created_at', 'updated_at'
            ],
            'face_encodings': [
                'id', 'user_id', 'encoding_vector', 'created_at'
            ]
        }
        
        # Verify each table and its columns
        for table_name, expected_columns in expected_tables.items():
            if table_name not in inspector.get_table_names():
                print(f"Error: Table '{table_name}' was not created!")
                return False
                
            actual_columns = [col['name'] for col in inspector.get_columns(table_name)]
            missing_columns = set(expected_columns) - set(actual_columns)
            
            if missing_columns:
                print(f"Error: Table '{table_name}' is missing columns: {missing_columns}")
                return False
        
        return True

def init_db():
    """Initialize the database with required tables"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Create Flask app
        app = create_app()
        
        # Verify database URL is set
        if not app.config['SQLALCHEMY_DATABASE_URI']:
            print("Error: DATABASE_URL environment variable is not set!")
            print("Please check your .env file and ensure DATABASE_URL is properly configured.")
            sys.exit(1)
        
        print("\nCreating database tables...")
        if create_tables(app):
            print("Database initialization successful!")
            print("\nCreated tables with the following structure:")
            print("\nTable: users")
            print("  - id (Integer, Primary Key)")
            print("  - name (String(100), Not Null)")
            print("  - created_at (DateTime)")
            print("  - updated_at (DateTime)")
            print("\nTable: face_encodings")
            print("  - id (Integer, Primary Key)")
            print("  - user_id (Integer, Foreign Key to users.id)")
            print("  - encoding_vector (LargeBinary, Not Null)")
            print("  - created_at (DateTime)")
        else:
            print("\nError: Database initialization failed!")
            sys.exit(1)
            
    except SQLAlchemyError as e:
        print("\nError: Database initialization failed!")
        print("SQLAlchemy Error:", str(e))
        print("\nCommon solutions:")
        print("1. Verify PostgreSQL is running")
        print("2. Check your DATABASE_URL in .env file")
        print("3. Ensure the database exists and is accessible")
        print("4. Verify your PostgreSQL username and password")
        sys.exit(1)
    except Exception as e:
        print("\nError: An unexpected error occurred!")
        print("Error details:", str(e))
        sys.exit(1)

def verify_environment():
    """Verify the environment is properly set up"""
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("Error: .env file not found!")
        print("Please create a .env file with your database configuration.")
        print("Example .env file:")
        print("DATABASE_URL=postgresql://username:password@localhost:5432/face_recognition")
        sys.exit(1)
    
    # Check if required environment variables are set
    required_vars = ['DATABASE_URL']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: Missing required environment variables:", ", ".join(missing_vars))
        sys.exit(1)

if __name__ == '__main__':
    print("Starting database initialization...")
    verify_environment()
    init_db() 