from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model = joblib.load('rf_model.pkl')

        height = request.form['heit']
        hand_len = request.form['hlen']
        foot_len = request.form['flen']

        mapping = {1: 'Male', 2: 'Female'}

        df = pd.DataFrame([[height, hand_len, foot_len]], columns=['height', 'hand_len', 'foot_len'])
        prediction = model.predict(df.values)[0]
        results = mapping[prediction]

        return render_template('index.html', results=results)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
