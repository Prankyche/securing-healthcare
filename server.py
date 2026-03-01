from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO
import random
import time
from traffic_simulator import generate_packet_result

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

@app.route("/")
def home():
    return render_template("index.html")

def stream_packets():
    while True:
        result = generate_packet_result()
        socketio.emit("new_packet", result)
        socketio.sleep(2)

@socketio.on("connect")
def connected():
    print("Dashboard connected")

if __name__ == "__main__":
    socketio.start_background_task(stream_packets)
    socketio.run(app, port=5000, debug=True, allow_unsafe_werkzeug=True)
