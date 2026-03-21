# server.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/data', methods=['GET'])
def get_data():
    response = {
        "message": "Hello from Flask!",
        "status": "success"
    }
    return jsonify(response)


@app.route('/number', methods=['GET'])
def get_number():
    response = {
        "message": "13",
        "status": "success"
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
