# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:02:44 2019

@author: Puchu
"""

import pandas as pd
file_handler = open("C:\\Users\\Puchu\\.spyder-py3\\diabetes.csv", "r") 
  
# creating a Pandas DataFrame 
# using read_csv function  
# that reads from a csv file. 
dataset = pd.read_csv(file_handler, sep = ",") 
  
# closing the file handler 
file_handler.close() 
  

X=dataset.iloc[:,0:8].values
y=dataset.iloc[:,8].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


from sklearn.linear_model import LogisticRegression, Perceptron

classifier = LogisticRegression(penalty='l2', dual=False, tol=0.0001,
                                                  C=1.0, fit_intercept=True, intercept_scaling=1,
                                                  class_weight=None, random_state=2, solver='liblinear',
                                                  max_iter=100, multi_class='ovr', verbose=2,
                                                  warm_start=True, n_jobs=-1)

classifier2 = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, max_iter=100,
                                      tol=None, shuffle=True, verbose=0, eta0=1.0, n_jobs=None,
                                      random_state=0, early_stopping=False, validation_fraction=0.1,
                                      n_iter_no_change=5, class_weight=None, warm_start=False,
                                      n_iter=None)
clf= classifier.fit(X_train,y_train)

clf2= classifier2.fit(X_train,y_train)

pred = clf.predict(X_test)
pred2 = clf2.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test,pred)
print ("confusion matrix of logistic regression:\n")
cm

cm2=confusion_matrix(y_test,pred2)
print ("confusion matrix of perceptron:\n")
cm2

ac=100*accuracy_score(y_test,pred)
print ("acurracy of logistic regression:\n")
ac

ac2=100*accuracy_score(y_test,pred2)
print ("accuracy of perceptron:\n")
ac2

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
objects = ('Perceptron','Logistic Regression')
y_pos = np.arange(len(objects))
performance = [ac2,ac]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('accuracy')
plt.title('Comparing model Accuracy')
 
plt.show()
if(ac > ac2):
    print("logistic regression is the superior model among the two model")
else:
    print("percetron is the superior model among the two model")