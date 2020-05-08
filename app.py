from pickle import UnpicklingError
from sklearn.model_selection import train_test_split as tts
from sklearn import metrics
import time
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from flask import Flask, request
#from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split as tts
from sklearn import metrics
import pickle
# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')

try:
    model = pickle.load(open('rf_plain.pkl', 'rb'))
except:
    UnpicklingError

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)


    output = round(prediction[0], 2)
    if (output == int(1)):
        k = "Have Diabaties"
    else:
        k = "Does'not have Diabaties"

    return render_template('index.html',prediction_text=k )
@app.route('/upload')
def upload():
    return render_template('upload.html')
@app.route('/uploader', methods=['GET','POST'])
def uploader():
    f = request.files['file']
    #f.save(f.filename)
    df1 = pd.read_csv(f)
    lt1 = time.ctime()
    xx=''
    s = ''
    for i in range(10, 19):
        s = s + lt1[i]

    #y1 = df1.Outcome
    y1 = df1.iloc[:,-1]
    x1 = df1.iloc[:,:8]
    #x1 = df1.drop('Outcome', axis=1)
    #x1_train, x1_test, y1_train, y1_test = tts(x1, y1, test_size=0.2)
   
    yhat = model.predict(x1)
    lt2=time.ctime()
    for i in range(10, 19):
        xx = xx+ lt2[i]



    accurac = (metrics.accuracy_score(y1, yhat))
    accuracy1 = round(accurac, 2)
    # return "hiii"
    return render_template('upload.html',acc=accuracy1)


#prediction_text='Employee Salary should be $ {}'.format(output)
if __name__ == "__main__":
    app.run(debug=True)
