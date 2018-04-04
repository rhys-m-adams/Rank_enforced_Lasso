# cvxopt_lasso
This code uses cvxopt to solve lasso linear regression. Because it uses cvxopt, it has some 
interesting variations you can do, such as enforcing the rank of coefficients, or
relaxing the lasso penalty on certain coefficients.

You'll need to install cvxopt and have the usual matplotlib, numpy, and scipy libraries
to run this code. I ran this code on Python 2.7.13 from anaconda (https://www.anaconda.com/download/). This used scipy=0.19.1, numpy=1.13.1, pandas=0.20.3. I installed the additional programs
```
conda install -y -c conda-forge cvxopt
```
Demo code can be found in lasso_demo_using_cvxopt.ipynb