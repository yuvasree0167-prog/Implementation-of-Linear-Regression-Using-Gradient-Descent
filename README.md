# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset
Read the CSV file and extract:
  Input feature → R&D Spend (X)
  Output → Profit (y)
  
2. Normalize the input feature X using mean and standard deviation to improve convergence of gradient descent.

3.Initialize parameters
Set initial values:
Slope m = 0
Intercept b = 0
Learning rate and number of iterations (epochs)

4.Predict output using:y=mx+b
y-intercept
x-intercept
Calculate error
Update m and b to reduce the error

5. Display results
Print final values of m and b and plot the graph showing actual data and regression line.

## Program:
Program to implement the linear regression using gradient descent.
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Startup.csv")
X = data['R&D Spend'].values
y = data['Profit'].values
X = (X - X.mean()) / X.std()
m = 0
b = 0
learning_rate = 0.01
epochs = 1000
n = len(X)
for i in range(epochs):
    y_pred = m * X + b
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    m = m - learning_rate * dm
    b = b - learning_rate * db
print("Slope (m):", m)
print("Intercept (b):", b)
y_pred = m * X + b
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")
plt.show()
```

Developed by: YUVASREE S

RegisterNumber:  212225230314

## Output:
<img width="844" height="613" alt="Screenshot 2026-04-22 110627" src="https://github.com/user-attachments/assets/c712a1b7-53b8-42cf-8c05-e3a5d7929b49" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
