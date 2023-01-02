
import json
from flask import Flask, request,jsonify



from flask import Flask

app = Flask(__name__)

@app.route('/predict/', methods=['GET', 'POST'])
def do_predict():
    data = json.loads(request.data)
    text = data.get("my_text",None)
    
    return jsonify({"fake":text})
    

if __name__ == '__main__':
    #print(predict("this text needs to be predicted"))
    app.run(host='0.0.0.0', port=5000)