from flask import Flask, render_template, request
from sklearn.externals import joblib
import os

app = Flask(__name__, static_url_path='/static/')


@app.route('/')
def form():
    return render_template('index.html')


@app.route('/predict_ride', methods=['POST', 'GET'])
def predict_ride():
    # get the parameters
    num_week = float(request.form['num_week'])
    weekday = float(request.form['weekday'])
    hour = float(request.form['hour'])
    pressure = float(request.form['pressure'])
    temp_celsius = float(request.form['temp_celsius'])
    humidity = float(request.form['humidity'])
    wind_speed = float(request.form['wind_speed'])

    # load the model and predict
    model = joblib.load('model/linear_regression.pkl')
    prediction = model.predict([[num_week,weekday,hour,pressure,temp_celsius,humidity,wind_speed]])
    predicted_ride = prediction.round(1)[0]

    return render_template('results.html',
                           num_week=int(num_week),
                           weekday=int(weekday),
                           hour=int(hour),
                           pressure=int(pressure),
                           temp_celsius=int(temp_celsius),
                           humidity=int(humidity),
                           wind_speed=int(wind_speed),
                           predicted_ride="{:,}".format(predicted_ride)
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
