from flask import Flask,render_template,request
import pickle
import numpy as np

model=pickle.load(open("fruit.pkl","rb"))

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('myform.html')
@app.route("/myform", methods=["POST"])
def home():
    mass=request.form['mass']
    hight=request.form['width']
    width=request.form['hight']
    color=request.form['color_score']
    arr=np.array([[mass,width,hight,color]])
    pred=model.predict(arr)
    return render_template('after.html',data=pred)



if __name__ == "__main__":
    app.run(debug=True)