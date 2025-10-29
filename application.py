from flask import Flask,request,jsonify,render_template
import pickle

from sklearn.preprocessing import StandardScaler

app= Flask(__name__)

ridge_modal1= pickle.load(open('modals/ridge.pkl','rb'))
standerd_scaler = pickle.load(open('modals/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET', 'POST'])
def predict_datapoint():
    if  request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        newdata_scaled= standerd_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_modal1.predict(newdata_scaled)

        return render_template('home.html',results= result[0])
    else :
        return render_template('home.html')
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5003, debug=True)



 