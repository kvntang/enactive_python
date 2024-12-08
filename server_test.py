from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api/process', methods=['POST'])
def process_data():
    data = request.json
    # expected to recieve {name:_, secret:_}
    # Extract the name from the request body
    name = data.get("name", "Guest")  # Default to "Guest" if name is not provided
    secret = data.get("secret", "no secret")  # Default to "Guest" if name is not provided

    # Customize the response with the extracted name
    result = {
        "message": f"Hello, {name}! Your secret is {secret}."
    }
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
