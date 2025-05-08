from flask import Flask
from config import Config
from routes import setup_routes

# Initialize Flask app
import os
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)
print("Template folder:", template_dir)

# Load configuration
app.config.from_object(Config)

# Setup routes
setup_routes(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)