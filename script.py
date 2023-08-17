
from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 5)
    #print(to_predict)
    loaded_model = pickle.load(open("C:\\Users\\Vivupadi\\Desktop\\House_price\\model\\price_model.pkl", 'rb'))
    print(type(loaded_model))
    result = loaded_model.predict(to_predict)
    return result[0]
 
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)              
        return render_template("result.html", prediction=result)
    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)