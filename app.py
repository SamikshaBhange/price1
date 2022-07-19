
from pandas import *
from flask import Flask, render_template, request
import joblib
import numpy
app=Flask(__name__)
model = joblib.load("training_files/smodel.pkl")
@app.route('/')
def index():
     return render_template("index.html")
@app.route('/pred',methods=["POST","GET"])
def pred():

               c=request.form.get("co_file")

               m=request.form.get("mo_file")
               y=request.form.get("yr_file")
               f=request.form.get("fu_file")
               km=request.form.get("km_file")
               print(c,m,y,f,km)
              #input_features = [i for i in request.form.values()]
              #nput_numpy = [numpy.array(input_features)]
              #prediction = model.predict(input_numpy)
              #return render_template("index.html", prediction_text="Predicted Salary is:{}".format(prediction[0]))

               prediction=model.predict(DataFrame([[m,c,y,km,f]],columns=["name", "company", "year", "kms_driven", "fuel_type"]))
               p=numpy.round(prediction,2)
               print(p)
               return render_template("index.html", prediction_text="Predicted Price is : {}".format(p[0]))
if __name__=="__main__":
    app.run(debug=True)