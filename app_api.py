from distutils.log import debug
from flask import Flask, request, url_for
import bp_prediction as bpp

import json
import numpy as np
import random

app = Flask(__name__)


# input to numpy array
@app.route('/', methods=['GET', 'POST'])
def home():
    request_made = request.get_json()
    return json.dumps({"status": "Ayurlite server Running ok!!"})

    
@app.route('/bp-prediction-ppg', methods=['GET', 'POST'])
def bp_prediction_ppg():
    request_made = request.get_json()
   
    input = np.asarray(list(map(float, json.loads(request_made["ppg"]))))


    print("input ::: ",input)
    tt = bpp.blood_pressure(input)
   

    return json.dumps({'type': tt})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000,debug=True)





