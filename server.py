from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api/process', methods=['POST'])
def process_data():
    data = request.json
    result = {"message": "Received", "data": data}
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
