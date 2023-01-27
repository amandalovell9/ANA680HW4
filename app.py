from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
app = Flask(__name__)
filename = 'file_cancer.pkl'
#model = pickle.load(open(filename, 'rb'))
clf2 = joblib.load(filename)
#model = joblib.load(filename)
@app.route('/')
def index(): 
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    meanradius = request.form['meanradius']
    meantexture = request.form['meantexture']
    meanperimeter = request.form['meanperimeter']
    meanarea = request.form['meanarea']

    
      
    pred = clf2.predict(np.array([[meanradius, meantexture, meanperimeter, meanarea]], dtype=float))
    print(pred)
    return render_template('index.html', predict=str(pred))


if __name__ == '__main__':
    app.run