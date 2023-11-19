# LinearRegression-from-scratch
Explain the 2 files code

Code is implementation of LinearRegression 
a file called LinearRegression is the Class Including Functions


First thing I created a "Class" which is called "LinearRegression" In it I wrote functions such as Fit, predic.
**class LinearRegression:**

in the class, I create __init__() function and passed to it lr "Learning rate", n_iters "Num_of_Iterations" and assign them

    **def __init__(self,lr=.001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None**



in the "Fit fun" and passed to its input as "X" and output as "y" 
first, I got num_of_smaples "Rows" and "Num of Features "Columns" from the shape of X

**"f(X)=WX+b"**
initial value for w and b be a zero so I created self.bias=0, self.weights=np.zeros "np stands for numpy" >>
an array of zeros, not one value because there are multiple features size of array =n_features

loops for  num of iterations then apply equations of Gradient descent for both W and  B
then apply changes to them simultaneously 






    **def fit(self,X,y):
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iters):
            y_pred=np.dot(X,self.weights)+self.bias


            dw=(1/n_samples)*np.dot(X.T,(y_pred-y))
            db=(1/n_samples)*np.sum(y_pred-y)


            self.weights=self.weights - self.lr*dw
            self.bias=self.bias -self.lr*db**




Then **"Predict fun"**
**y_pred=WX+b**
and return y_pred


    def predict(self,X):
        y_pred=np.dot(X,self.weights)+self.bias
        return y_pred



**"Test file"**

Importing python libraries 
   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn import datasets
   import matplotlib.pyplot as plt

it is not a python library it is the class which i create it above 
from LinearRegresion import LinearRegression
from LinearRegresion(file name) import LinearRegression (class name)






