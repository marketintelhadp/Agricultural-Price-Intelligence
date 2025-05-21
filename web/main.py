# main.py - Flask app launcher
from flask import Flask
from routes import setup_routes
from realtime_routes import realtime_bp
import os
import sys

# Ensure 'web' directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set template folder
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)

# Setup main forecast and real-time dashboard routes
setup_routes(app)
app.register_blueprint(realtime_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
