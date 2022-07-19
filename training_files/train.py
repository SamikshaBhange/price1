import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn import *
import joblib
from pandas import *

d=read_csv("C:/Users/s.ravindra.bhange/Desktop/jyputr/car.csv")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(d[["name", "company", "year", "kms_driven", "fuel_type"]], d["Price"], test_size=0.2)
ohe=OneHotEncoder()
ohe.fit(d[["name", "company", "fuel_type"]])
reg=linear_model.LinearRegression()
column=make_column_transformer((OneHotEncoder(categories=ohe.categories_), ["name", "company", "fuel_type"]), remainder="passthrough")
pipe=make_pipeline(column, reg)
pipe.fit(x_train, y_train)
op=pipe.predict(x_test)
print(op)
joblib.dump(pipe,"smodel.pkl")
model = joblib.load("smodel.pkl")
p = model.predict(DataFrame([["Hyundai Santro Xing","Hyundai",2007,450,"Petrol"]],columns=["name", "company", "year", "kms_driven", "fuel_type"]))
print(p)
