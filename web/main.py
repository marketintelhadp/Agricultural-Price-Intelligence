# web/main.py

from flask import Flask
from routes import setup_routes
from realtime_routes import realtime_bp
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)

setup_routes(app)
app.register_blueprint(realtime_bp)

application = app  # ← for Gunicorn

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
