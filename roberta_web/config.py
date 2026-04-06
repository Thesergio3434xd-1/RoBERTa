import os


class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "roberta-web-secret-key")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
    HISTORY_FOLDER = os.path.join(os.path.dirname(__file__), "data", "historial")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")