import sqlite3
import hashlib
import datetime
import MySQLdb
import numpy as np
import pandas as pd
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
warnings.simplefilter('ignore')
from flask import session
from random import randint

def db_connect():
    _conn = MySQLdb.connect(host="localhost", user="root",
                            passwd="root", db="nafld")
    c = _conn.cursor()

    return c, _conn



# -------------------------------Registration-----------------------------------------------------------------



def user_reg(username, password,  dob,mobile, email):
    try:
        c, conn = db_connect()
        print(username, password,  dob,mobile, email)
        
        id="0"
        j = c.execute("insert into user (id,username, password,  dob,mobile, email) values ('"+id+"','"+username +
                      "','"+password+"','"+dob+"','"+mobile+"','"+email+"')")
        conn.commit()
        conn.close()
        print(j)
        return j
    except Exception as e:
        print(e)
        return(str(e))
  



# -------------------------------Registration End-----------------------------------------------------------------
# -------------------------------Loginact Start-----------------------------------------------------------------

def admin_loginact(username, password):
    try:
        c, conn = db_connect()
        j = c.execute("select * from admin where username='" +
                      username+"' and password='"+password+"'")
        c.fetchall()
        conn.close()
        return j
    except Exception as e:
        return(str(e))

def user_loginact(username, password):
    try:
        c, conn = db_connect()
        j = c.execute("select * from user where username='" +
                      username+"' and password='"+password+"'")
        c.fetchall()
        conn.close()
        return j
    except Exception as e:
        return(str(e))

def view_emp():
    c, conn = db_connect()
    c.execute("select * from user")
    result = c.fetchall()
    conn.close()
    print("result")
    return result


#---------------------------------------------------main code------------------------------------------------------------------
def Datapreparation():

    df = pd.read_excel('NAFLD.xlsx')
    df_X = df.iloc[:, 1:-11]
    sig_fib_y = pd.DataFrame(df.iloc[:, -5])
    adv_fib_y = pd.DataFrame(df.iloc[:, -4])
    df_X.rename(columns={'Smoking Status (Never Smoked=1, Left Smoking=2, Smoking=3)': 'Smoking Status'}, inplace=True)
    sig_fib_y.rename(columns={'Significant Fibrosis (No=0, Yes=1) (If Fibrosis 2 and above, there is Significant Fibrosis)': 'Significant Fibrosis'}, inplace=True)
    adv_fib_y.rename(columns={'Advanced Fibrosis (No=0, Yes=1) (If Fibrosis is 3 and above, there is Advanced Fibrosis)': 'Advanced Fibrosis'}, inplace=True)


    nan_columns = df_X.columns[df_X.isna().any()].tolist()
    X_columns_1 = df_X.isna().sum().sort_values(ascending=False)[-15:].index   # features with no missing values
    X_columns_2 = df_X.isna().sum().sort_values(ascending=False)[15:].index    # features <= 25% missing values threshold
    X_baseline_1 = df_X[X_columns_1]   # Data Frame of features with no missing values
    X_baseline_2 = df_X[X_columns_2]   # Data Frame of features with <= 25% missing values threshold

    features=X_baseline_1.head()
    print(features)
    labels  = adv_fib_y.head()
    print(labels)
    return features,labels

def calldata():

    df = pd.read_excel('NAFLD.xlsx')
    df_X = df.iloc[:, 1:-11]
    sig_fib_y = pd.DataFrame(df.iloc[:, -5])
    adv_fib_y = pd.DataFrame(df.iloc[:, -4])
    df_X.rename(columns={'Smoking Status (Never Smoked=1, Left Smoking=2, Smoking=3)': 'Smoking Status'}, inplace=True)
    sig_fib_y.rename(columns={'Significant Fibrosis (No=0, Yes=1) (If Fibrosis 2 and above, there is Significant Fibrosis)': 'Significant Fibrosis'}, inplace=True)
    adv_fib_y.rename(columns={'Advanced Fibrosis (No=0, Yes=1) (If Fibrosis is 3 and above, there is Advanced Fibrosis)': 'Advanced Fibrosis'}, inplace=True)


    nan_columns = df_X.columns[df_X.isna().any()].tolist()
    X_columns_1 = df_X.isna().sum().sort_values(ascending=False)[-15:].index   # features with no missing values
    X_columns_2 = df_X.isna().sum().sort_values(ascending=False)[15:].index    # features <= 25% missing values threshold
    X_baseline_1 = df_X[X_columns_1]   # Data Frame of features with no missing values
    X_baseline_2 = df_X[X_columns_2]   # Data Frame of features with <= 25% missing values threshold
    return X_baseline_1,X_baseline_2,X_columns_1,X_columns_2,adv_fib_y,sig_fib_y

def  Normalization():
    sc = StandardScaler()
    X_baseline_1,X_baseline_2,X_columns_1,X_columns_2,adv_fib_y,sig_fib_y = calldata()
    std_baseline_1 = sc.fit_transform(X_baseline_1)
    std_baseline_2 = sc.fit_transform(X_baseline_2)
    X_baseline_1 = pd.DataFrame(std_baseline_1, dtype=float, columns=X_columns_1)
    X_baseline_2 = pd.DataFrame(std_baseline_2, dtype=float, columns=X_columns_2)

    ndata=X_baseline_1.head(10)
    print(ndata)
    return X_baseline_1,X_baseline_2,adv_fib_y,sig_fib_y

def  Normalization1():
    sc = StandardScaler()
    X_baseline_1,X_baseline_2,X_columns_1,X_columns_2,adv_fib_y,sig_fib_y = calldata()
    std_baseline_1 = sc.fit_transform(X_baseline_1)
    std_baseline_2 = sc.fit_transform(X_baseline_2)
    X_baseline_1 = pd.DataFrame(std_baseline_1, dtype=float, columns=X_columns_1)
    X_baseline_2 = pd.DataFrame(std_baseline_2, dtype=float, columns=X_columns_2)

    ndata=X_baseline_1.head(10)
    print(ndata)
    return ndata

def graphs():
    X_baseline_1,X_baseline_2,adv_fib_y,sig_fib_y= Normalization()
    fig = plt.figure(figsize = (15,15))
    ax = fig.gca()
    X_baseline_1.hist(ax=ax)
    #plt.show()
    plt.savefig("static/img/graphs.png")

    temp = X_baseline_1.join(adv_fib_y)
    correlation = temp.corr()[adv_fib_y.columns].abs()
    baseline_1_adv = correlation.sort_values(by=[adv_fib_y.columns[0]], ascending=False)[1:6]   # top 5 pos_corr (features with no missing values and adv_fib)
    #baseline_1_adv

    temp = X_baseline_1.join(sig_fib_y)
    correlation = temp.corr()[sig_fib_y.columns].abs()
    baseline_1_sig = correlation.sort_values(by=[sig_fib_y.columns[0]], ascending=False)[1:6]   # top 5 abs_corr (features with no missing values and sig_fib)
    baseline_1_sig
    return temp

def ModelCreation():
    X_baseline_1,X_baseline_2,adv_fib_y,sig_fib_y= Normalization()
    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_baseline_1, adv_fib_y, random_state=0, test_size=0.2, stratify=adv_fib_y)
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_baseline_2, adv_fib_y, random_state=0, test_size=0.2, stratify=adv_fib_y)

    return X_train_0, X_test_0, y_train_0, y_test_0

def algorithm_accuracy():
    X_train_0, X_test_0, y_train_0, y_test_0 =ModelCreation()
    X_baseline_1,X_baseline_2,X_columns_1,X_columns_2,adv_fib_y,sig_fib_y = calldata()
    clf = SVC()
    clf.fit(X_train_0, y_train_0)
    output = clf.predict(X_test_0)
    print(output)
    print ('support vecotr')
    
    RFC_Classifier = RandomForestClassifier(max_depth=40)
    RFC_Classifier.fit(X_train_0, y_train_0)
    output = RFC_Classifier.predict(X_test_0)
 
    print(output)
    print ('RF Classifier run')

    decission_tree = tree.DecisionTreeClassifier()   # empty model of the decision tree
    decission_tree = decission_tree.fit(X_train_0, y_train_0)
    output = decission_tree.predict(X_test_0)
    print ('decission tree')


    gnb = GaussianNB()
    gnb=gnb.fit(X_train_0, y_train_0)
    output = gnb.predict(X_test_0)
    print ('Navie bayes')

    models = []
    models.append(('Random Forest Classifier', RFC_Classifier))
    #models.append(('Decision Tree Classifier', decission_tree))
    #models.append(('Support Vector Classifier',clf))
    #models.append(('Support Vector Classifier',gnb))
    
    results = []
    names = []
    
    Xpred = RFC_Classifier.predict(X_train_0)
    print("predict")
    #print(Xpred[0:10])
    scores = cross_val_score(RFC_Classifier, X_train_0, y_train_0, cv=10)
    results.append(scores)
    names.append(RFC_Classifier)
    accuracy = metrics.accuracy_score(y_train_0, Xpred)
    confusion_matrix = metrics.confusion_matrix(y_train_0, Xpred)
    classification = metrics.classification_report(y_train_0, Xpred)
    print()
    print('============================== {} Model Evaluation ==============================')
    print()
        
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
       
    plt.figure(figsize=(5, 7))
         
    ax = sns.distplot(adv_fib_y, hist=False, color="r", label="Actual Value",kde_kws={'bw':0.1})
    sns.distplot(Xpred, hist=False, color="b", label="predicted Values" , ax=ax)
    plt.title('Actual vs predicted Values ')
    #plt.show()
    plt.savefig("static/img/algaccuracy.png")


    #fig = plt.figure()
    #print("sssssssssssss")
    #fig.suptitle('Algorithm Comparison')
    #ax = fig.add_subplot(111)
    #plt.boxplot(results)
    #ax.set_xticklabels(names)
    #plt.show()
    #plt.savefig("static/img/algcomp.png")
    return accuracy 

def predict(request):

    X_train_0, X_test_0, y_train_0, y_test_0 =ModelCreation()
        
    RFC_Classifier = RandomForestClassifier(max_depth=40)
    RFC_Classifier.fit(X_train_0, y_train_0)
    X_baseline_1,X_baseline_2,X_columns_1,X_columns_2,adv_fib_y,sig_fib_y = calldata()
   
    sc = StandardScaler()
    string = request
    #string='72 1 154 73 30.78 120 0 4.4 0 0 0 17.0 18 170 55'
    #string='80 2 170 101 34.95 120 1 5.1 1 1 1 51.0 74 190 53'

    data = string.split()
    print(data)
    print("Type:", type(data))
    print("Length:", len(data))

    for i in range(15):
        print(data[i])
    data = [float(x.strip()) for x in data]

    for i in range(15):
        print(data[i])

    data_np = np.asarray(data, dtype = float)
    data_np = data_np.reshape(1,-1)
    #out, acc, t = predict_svm(clf, data_np)
    std_baseline_1 = sc.fit_transform(X_baseline_1) 

    inp = sc.transform(data_np)
    print(inp)
    output = RFC_Classifier.predict(inp)    
    print(output)
    return output
if __name__ == "__main__":
    print(db_connect())
