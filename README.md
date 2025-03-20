## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

dataset.info()

#assigning hours to x & scores to y
x = dataset.iloc[:,:-1].values
print(x)
y = dataset.iloc[:,-1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

print(X_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(Y_train.shape)

mse=mean_squared_error(y_test,y_pred) 
print('MSE = ',mse) 
mae=mean_absolute_error(y_test,y_pred) 
print('MAE = ',mae) 
rmse=np.sqrt(mse) 
print('RMSE = ',rmse)

plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,y_pred,color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

a=np.array([[13]])
y_pred1 = reg.predict(a)
print(y_pred1)
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: AMIRTHAVARSHINI V
RegisterNumber: 212223040014


## Output:
![Screenshot 2025-03-20 100801](https://github.com/user-attachments/assets/73062704-757d-4622-ab90-f993e2a798a7)
![Screenshot 2025-03-20 100816](https://github.com/user-attachments/assets/c48f3be0-6d7f-4031-a968-22a2d5b626ea)
![Screenshot 2025-03-20 100830](https://github.com/user-attachments/assets/334d6cdd-f0a3-4ef9-bc49-2b80e87ce929)
![Screenshot 2025-03-20 100857](https://github.com/user-attachments/assets/ee357e92-2b63-400a-8358-3dcefcdda3d9)
![Screenshot 2025-03-20 100912](https://github.com/user-attachments/assets/e4a816c0-b5cb-4dd2-935b-05552ca3dc4d)
![Screenshot 2025-03-20 100926](https://github.com/user-attachments/assets/338b3581-9309-49b7-aa4c-163ad36c0e11)
![Screenshot 2025-03-20 100939](https://github.com/user-attachments/assets/1043e0f7-4fba-44e0-ae27-b25de45ac7f3)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
