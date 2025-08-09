from flask import Flask, request, jsonify, render_template, Response
import json
import threading
import time
import queue
from app.logger import logger
from app.sinhala_logger import get_sinhala_logger

app = Flask(__name__, template_folder='../templates')
sinhala_logger = get_sinhala_logger()

@app.route('/')
def index():
    sinhala_logger.info("මුල් පිටුව ලොඩ් වේ")
    return render_template('index.html')

@app.route('/api/chat-stream', methods=['POST'])
def chat_stream():
    """Stream chat with Sinhala logging"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        # Log in Sinhala
        sinhala_logger.log_request(data, user_message)
        
        def generate():
            # Your existing streaming logic here
            sinhala_logger.info("ප්‍රතිචාර ධාරාව ආරම්භ වේ")
            yield f"data: {json.dumps({'content': 'ප්‍රක්‍රියාකරණය ආරම්භ වේ...'})}\n\n"
            
            # Add your actual processing logic here
            time.sleep(1)
            
            sinhala_logger.info("ප්‍රතිචාර ධාරාව අවසන් වේ")
            yield f"data: {json.dumps({'content': 'සම්පූර්ණයි!'})}\n\n"
        
        return Response(generate(), mimetype='text/plain')
        
    except Exception as e:
        sinhala_logger.log_error_sinhala(e, "chat stream endpoint")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
        return jsonify({"error": "දෝෂයක් සිදුවිය"}), 500

if __name__ == '__main__':
    sinhala_logger.info("Flask යෙදවුම ආරම්භ වේ")
    app.run(host='0.0.0.0', port=5000, debug=True)