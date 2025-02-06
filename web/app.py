from flask import Flask
from config import Config
from routes import setup_routes

# Initialize Flask app
app = Flask(__name__, template_folder='static/templates')

# Load configuration
app.config.from_object(Config)

# Setup routes
setup_routes(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
