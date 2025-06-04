from flask import Flask
from web.routes import setup_routes, mydash_bp
from web.realtime_routes import realtime_bp
import os
from dotenv import load_dotenv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load environment variables from .env
load_dotenv()

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
