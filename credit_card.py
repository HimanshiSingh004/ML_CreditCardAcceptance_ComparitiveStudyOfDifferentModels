import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics 
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import grid_search
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import pipeline
from sklearn import datasets
from sklearn import tree
import matplotlib.pyplot as plt

def printscore(prediction,actual,algo):
    print("ALGO : ",algo)
    print("confusion matrix : ")
    print(metrics.confusion_matrix(actual,prediction))
    print("Accuracy score : ",metrics.accuracy_score(actual,prediction))
    print("precision score : ",metrics.precision_score(actual,prediction))
    print("recall score : ",metrics.recall_score(actual,prediction))
    print("f1 score : ",metrics.f1_score(actual,prediction))
    print("AUC score : ",metrics.roc_auc_score(actual,prediction))
    
os.chdir("f:/datafiles")  
cc=pd.read_csv("credit_card.csv")
cc.head()
cc.columns.values
leenc=preprocessing.LabelEncoder()   
card_enc=leenc.fit_transform(cc["card"])
owner_enc=leenc.fit_transform(cc["owner"])
selfemp_enc=leenc.fit_transform(cc["selfemp"])
enc_cc=np.c_[card_enc,owner_enc,selfemp_enc]
df1=pd.DataFrame(enc_cc,columns=["card_","owner_","selfemp_"])
df2=pd.DataFrame(cc)
cc1=pd.concat([df1,df2],axis=1)
cc2=cc1.drop(["card","owner","selfemp"],axis=1)
cc2.columns.values
cc2.head()
x=cc2[['owner_','selfemp_','reports','age','income','share','expenditure','dependents','months','majorcards','active']]
y=cc2['card_']
xtrain,xtest,ytrain,ytest=model_selection.train_test_split(x,y,test_size=.25,random_state=42) 

linmod=linear_model.LinearRegression()  
linmod.fit(xtrain,ytrain)
print("Intercept : ",linmod.intercept_)
print("Coefficients : ",pd.DataFrame(linmod.coef_))
predicted1=linmod.predict(xtest)
rms_error=np.sqrt(np.average((predicted1-ytest)**2))
print("Root mean squared error is : ",rms_error)

model=naive_bayes.MultinomialNB()   
model.fit(xtrain,ytrain)
model.classes_
predicted2=model.predict(xtest)

knnclf=neighbors.KNeighborsClassifier() 
param_grid={"n_neighbors":[3,5,7,9,11],"weights":["uniform"]}
grid=grid_search.GridSearchCV(estimator=knnclf,param_grid=param_grid,scoring='roc_auc')
grid_fit=grid.fit(xtrain,ytrain)
predicted3=grid_fit.predict(xtest)

treeclf=tree.DecisionTreeClassifier(criterion="entropy",min_samples_leaf=10)
treeclf.fit(xtrain,ytrain)
predicted4=treeclf.predict(xtest)
tree.export_graphviz(treeclf,out_file='f:/pic/credit.dot',feature_names=cc2.columns.values[:11],class_names=["yes","No"],rounded=True,filled=True)

printscore(predicted2,ytest,"NAIVE BAYES")
printscore(predicted3,ytest,"K NEAREST NEIGHBOR")
printscore(predicted4,ytest,"DECISION TREE")
