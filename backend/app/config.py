import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/smart_mix_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
