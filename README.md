# Face Recognition Workshop

A real-time face recognition system built with Python, PyTorch, and Flask. This project demonstrates how to implement face detection and recognition using deep learning models, with a web interface for real-time video processing and user management.

## Features

- Real-time face detection using MTCNN
- Face recognition using FaceNet (InceptionResnetV1)
- Live video streaming with WebRTC
- User management interface for enrolling faces
- PostgreSQL database for storing face encodings
- Docker support for easy deployment

## Tech Stack

- Python 3.11
- PyTorch with facenet-pytorch
- OpenCV for image processing
- Flask web framework
- PostgreSQL database
- WebRTC for video streaming
- Bootstrap for UI

## Prerequisites

- Python 3.11 or higher
- PostgreSQL database
- Webcam for live video
- Modern web browser with WebRTC support

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ekkirinaldi/workshop-face-recognition.git
   cd workshop-face-recognition
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up the database:
   ```bash
   createdb face_recognition
   PYTHONPATH=. python scripts/init_db.py
   ```

4. Create a `.env` file:
   ```
   FLASK_APP=app
   FLASK_ENV=development
   DATABASE_URL=postgresql:///face_recognition
   SECRET_KEY=your-secret-key-here
   ```

## Running the Application

### Development Mode

```bash
python wsgi.py
```

The application will be available at:
- Main interface: http://localhost:5000
- Admin interface: http://localhost:5000/admin

### Production Mode (Docker)

```bash
docker-compose up --build
```

## Usage

1. Access the admin interface at `/admin`
2. Add users and their face images
3. Go to the main interface to see live face recognition
4. The system will:
   - Show green boxes around unrecognized faces
   - Show blue boxes with names around recognized faces

## Project Structure

```
face-recognition-workshop/
├── app/
│   ├── models/
│   │   ├── database.py
│   │   └── face_recognition.py
│   ├── routes/
│   │   ├── main.py
│   │   └── admin.py
│   ├── static/
│   ├── templates/
│   └── config.py
├── scripts/
│   └── init_db.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## API Endpoints

- `POST /api/recognize`: Face recognition endpoint
  - Input: Base64 encoded image
  - Output: List of detected faces with recognition results

- `GET /admin/users`: List all users
- `POST /admin/users`: Create new user with face
- `POST /admin/users/<id>/faces`: Add face to existing user
- `DELETE /admin/users/<id>`: Delete user

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 