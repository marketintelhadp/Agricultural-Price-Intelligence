from flask import Flask
from routes import setup_routes
from realtime_routes import realtime_bp
from routes import mydash_bp  # make sure this file exists

import os
import sys

# Ensure 'web' directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask
app = Flask(__name__, template_folder='templates')

# Secret key is required for flash messages
app.secret_key = 'dfbbirbgiu348fdugkjfg;jbskjgbkjdsbfgjbberii3'

# Register routes
setup_routes(app)
app.register_blueprint(realtime_bp)
app.register_blueprint(mydash_bp)  # ✅ Register the mydash blueprint

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
