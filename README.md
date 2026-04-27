# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


data = {
    'Hours': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7],
    'Scores': [21, 47, 27, 75, 30, 20, 88, 60, 81, 25]
}

df = pd.DataFrame(data)

print(df)

X = df[['Hours']].values
Y = df['Scores'].values

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=1/3, random_state=0)


reg = LinearRegression()
reg.fit(Xtrain, Ytrain)


Ypred = reg.predict(Xtest)

print("Predicted Values:", Ypred)


plt.scatter(Xtrain, Ytrain, color='orange')
plt.plot(Xtrain, reg.predict(Xtrain), color='red')
plt.title("Training Set")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


plt.scatter(Xtest, Ytest, color='blue')
plt.plot(Xtest, reg.predict(Xtest), color='green')
plt.title("Test Set")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


mse = mean_squared_error(Ytest, Ypred)
mae = mean_absolute_error(Ytest, Ypred)
rmse = np.sqrt(mse)

print("MSE :", mse)
print("MAE :", mae)
print("RMSE :", rmse)
Developed by: Keerthika R
RegisterNumber: 212225040187 
*/
```

## Output:
<img width="763" height="270" alt="image" src="https://github.com/user-attachments/assets/3903fb6b-cbaf-4c4e-b419-5b285f1790c5" />
<img width="823" height="600" alt="image" src="https://github.com/user-attachments/assets/af2773af-514f-4b39-a604-bfb5a3b7fe4c" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
