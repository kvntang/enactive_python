from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api/process', methods=['POST'])
def process_data():

    data = request.json #incoming data

    # expected to recieve {type:_ ; 
    #                      steps:_; 
    #                      prompt_word:_ ; 
    #                      original_image:_}


    # noise operation


    # denoise/diffusion operation

    # new_image = blablablah
    
    # Customize the response with the extracted name
    # send back the "new_image"
    result = {
        "message": f"Hello."
    }
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001) #this port can be editted
