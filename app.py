import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import statsmodels.api as sm

app = Flask(__name__)
model = pickle.load(open('model2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    seniority = int(request.form["Seniority"])
    home = int(request.form["Home"])
    time = int(request.form["Time"])

    marital=int(request.form["Marital"])
    records=int(request.form["Records"])
    job=int(request.form["Job"])
    expenses=int(request.form["Expenses"])
    income=int(request.form["Income"])

    debt=int(request.form["Debt"])
    amount=int(request.form["Amount"])
    price=int(request.form["Price"])
    finrat = round((amount/price)*100,6 )
    savings = round((income-expenses-(debt/100))/(amount/time),6 )
    features=[1.0]
    features.extend([seniority, home, time, marital, records, job, expenses, income , debt, amount, price, savings])
    
    final_features=[np.array(features)]
    prediction = model.predict(sm.add_constant(final_features))

    output = round(prediction[0], 4)
    if output > 0.5:
        return render_template('index.html', prediction_text='According to our analysis considering the data provided by you, you seem to be a reliable customer to us with customer credit score: {} percent'.format(output*100))
    else:
        return render_template('index.html', prediction_text='Sorry to inform you but the bank won''t be able to extend you the loan as your customer credit score ({} percent) is too low currently.'.format(output*100))

if __name__ == "__main__":
    app.run(debug=True)