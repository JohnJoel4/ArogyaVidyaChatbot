# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_engine import setup_rag_engine, get_response

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS to allow requests from any origin (your future frontend)
CORS(app)

# --- APPLICATION STARTUP ---
# Set up the RAG engine once when the server starts
with app.app_context():
    setup_rag_engine()

# --- API ENDPOINT ---
@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handles chat requests from the user.
    """
    # Get the user's message from the request body
    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Get the response from our RAG engine
        bot_response = get_response(user_message)
        return jsonify({"reply": bot_response})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

# --- RUN THE APPLICATION ---
if __name__ == '__main__':
    # Runs the server on http://127.0.0.1:5000
    app.run(debug=True, port=5000)