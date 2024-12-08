from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api/process', methods=['POST'])
def process_data():

    data = request.json #incoming data

    # expected to recieve {type:_ string; 
    #                      steps:_string; 
    #                      prompt_word:_ string; 
    #                      original_image:_string}

    # 1. convert base64 to img

    # 2.1 noise operation

    # 2.2 denoise/diffusion operation

    # 3. new_image = blablablah
    
    # 4. send back the "new_image"

    result = {
        "message": f"Hello."
    }
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001) #this port can be editted
