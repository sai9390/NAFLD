import numpy as np
import pandas as pd
import pandas as pandas
import os
import matplotlib.pyplot as plt
from impyute.imputation.cs import fast_knn
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
import numpy as np
from sklearn.model_selection import validation_curve 
import seaborn as sns
from sklearn.model_selection import cross_val_score
import warnings
import numpy as np
from sklearn.model_selection import validation_curve 
import seaborn as sns
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash
from database import db_connect,  admin_loginact,user_reg, user_loginact,view_emp
from database import Datapreparation,Normalization1,graphs,algorithm_accuracy,predict
warnings.simplefilter('ignore')

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route("/")
def FUN_root():
    return render_template("index.html")

@app.route("/index.html")
@app.route("/logout")
def index():
    return render_template("index.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/user")
def user():
    return render_template("user.html")

@app.route("/userreg")
def userreg():
    return render_template("userreg.html")

@app.route("/adminhome")
def adminhome():
    return render_template("adminhome.html")

@app.route("/userhome")
def userhome():
    return render_template("userhome.html")

@app.route("/viewemp")
def viewemp():
    data = view_emp()
    return render_template("viewemp.html",empdetails = data)


@app.route("/analyze")
def analyze():
    return render_template("analyze.html")

@app.route("/finalgraph")
def finalgraph():
    return render_template("finalgraph.html")

@app.route("/predict")
def predict1():
    return render_template("predict.html")

@app.route("/view")
def view():
    data=pandas.read_excel(r'C:\Users\madas\OneDrive\Documents\mini project\non-alholic\NAFLD\SOURCE CODE\NAFLD.xlsx')
    peek=data.head(30)
    return render_template("analyze1.html",tables=[peek.to_html(classes='data')])

@app.route("/userreg", methods = ['GET','POST'])
def userreg1():
   if request.method == 'POST':      
      status = user_reg(request.form['username'],request.form['password'],request.form['dob'],request.form['mobile'],request.form['email'])
      if status == 1:
       return render_template("user.html",m1="sucess")
      else:
       return render_template("userreg.html",m1="failed")

@app.route("/adminlogact", methods=['GET', 'POST'])       
def adminlogact():
    if request.method == 'POST':
        status = admin_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1:
            session['username'] = request.form['username']
            return render_template("adminhome.html", m1="sucess")
        else:
            return render_template("admin.html", m1="Login Failed")

@app.route("/userlogact", methods=['GET', 'POST'])       
def userlogact():
        if request.method == 'POST':
           status = user_loginact(request.form['username'], request.form['password'])
           print(status)
        if status == 1:
            session['username'] = request.form['username']
            return render_template("userhome.html", m1="sucess")
        else:
            return render_template("user.html", m1="Login Failed")
@app.route("/datapreparation")
def Datapreparation1():
    labels,features = Datapreparation()
    return render_template("datapreparation.html",tables=[labels.to_html(classes='data')],tables1=[features.to_html(classes='data')])





    
@app.route("/normaliztion1")
def  Normalization2():
    abc = Normalization1()
    return render_template("normalization.html",tables=[abc.to_html(classes='data')])
    

@app.route("/graphs")
def graph():
    gra = graphs()
    return render_template("viewgraph.html")


    
@app.route("/result")
def result():
    accuracy = algorithm_accuracy()
    return render_template("result.html",result = accuracy)


@app.route("/next", methods = ['GET','POST'])
def next1():
    output = predict(request.form['request'])
    return render_template("predicted.html", data1 = output)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)