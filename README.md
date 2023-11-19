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

it is not a Python library it is the class that i create above 
         from LinearRegresion import LinearRegression
from LinearRegresion(file name) import LinearRegression (class name)


I imported a built Dataset and passed it its parameters
then used "X_train, X_test,y_train,y_test=train_test_split"Built-in fun from sklearn.model_selection
then used "X_train,X_test,y_train,y_test=train_test_split" here it splits data 80% :20%
80% for training
20% for Test
random_state >>that fixed Data that the model will use it


      X,y=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
      X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=1234)


this will visualize raw Data Distribution

      fig=plt.figure(figsize=(8,6))
      plt.scatter(X[:,0],y,color="b",s=30)
      plt.show()


"reg" is an object from the class "LinearRegression()"

      reg=LinearRegression()
      reg.fit(X_test,y_test)
      y_prediction=reg.predict(X_test)


this is a function for calculating Mean Square error

      def MSE (y_test,y_prediction):
          return np.mean((y_test-y_prediction)**2)


Visualization Data After fitting and drawing the best-fit line


      y_pred_line=reg.predict(X)
      cmap = plt.get_cmap('viridis')
      fig=plt.figure(figsize=(8,6))
      m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
      m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
      plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
      plt.show() 





