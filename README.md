# LinearRegression-from-scratch
Explain the 2 files code

Code is implemantion of LinearRegression 
file called LinearRegression is the Class Incliuding Functions


First thing i created a "Class" which is called "LinearRegression" in it i wrote functions such as Fit , predic.
**class LinearRegression:**

in the class i create __init__() function and passed to it lr "Learning rate", n_iters "Num_of_Iterations" and assisn them

    **def __init__(self,lr=.001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None**



in the "Fit fun"and passed to it input as "X" and output as "y" 
first i got num_of_smaples "Rows" amd "Num of Features "Columns" from shape of X

**"WX+b"**
intale value for w and b be a zero so i created self.bias=0, self.weights=np.zeros "np stands for numpy" >>
array of zeros not a one value because there are multiple features size of array =n_features






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


