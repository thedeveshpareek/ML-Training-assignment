import pandas as pd
import numpy as np
from flask import Flask,request,jsonify,render_template
from recommend_movie import movie
import pickle
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
DtreeModel=pickle.load(open('Dtree_model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predicts')
def predicts():
    return render_template('prediction.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_featurs=[float(x) for x in request.form.values()]
    final_featurs=[np.array(int_featurs)]
    prediction=model.predict(final_featurs)
    

    output=round(prediction[0],2)

    return render_template('prediction.html', prediction_text='Prediction of model $ {}'.format(output))
@app.route('/predict_api',methods=['GET', 'POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


@app.route('/dtpredict',methods=['GET', 'POST'])
def dtpredict():
    '''
    For rendering results on HTML GUI
    '''
    int_featurs=[float(x) for x in request.form.values()]
    final_featurs=[np.array(int_featurs)]
    prediction=DtreeModel.predict(final_featurs)
    

    output=round(prediction[0],2)

    return render_template('prediction.html', prediction_text='Prediction of model $ {}'.format(output))
@app.route('/dtpredict_api',methods=['GET', 'POST'])
def dtpredict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



@app.route('/reco',methods=['GET','POST'])
def reco():
    df=pd.DataFrame()
    if request.method=='POST':
        df=pd.DataFrame(movie.recommend_movie(request.form['movieName']))
    return render_template('recommend.html',tables=[df.to_html(classes='data')], titles=df.columns.values) 
@app.route('/recommend_api',methods=['GET', 'POST'])
def recommend_api():
    res = movie.recommend_movie(request.args.get('title'))
    return jsonify(res)

if __name__ == "__main__":
    app.run()
