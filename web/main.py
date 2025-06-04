from flask import Flask
from routes import setup_routes
from realtime_routes import realtime_bp
from routes import mydash_bp
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Ensure 'web' directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask
app = Flask(__name__, template_folder='templates')

# Use secret key from .env
app.secret_key = os.getenv('SECRET_KEY', 'fallback_key')

# Register routes
setup_routes(app)
app.register_blueprint(realtime_bp)
app.register_blueprint(mydash_bp)

if __name__ == '__main__':
    try:
        port = int(os.environ.get("PORT", 5000))
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        import logging
        logging.exception("App failed to start due to:")
