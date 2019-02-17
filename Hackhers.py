##CREATE USER DATA
import csv
import pandas as pd 
import numpy as np
name='Number_of_Jobs_Submitted.csv'
name2='Number_of_Jobs_Running.csv'
name3='Job_Size_Max_(Core_Count).csv'
name4='Job_Size_Min_(Core_Count).csv'
name5='Job_Size_Per_Job_(Core_Count).csv'
name6='Number_of_Jobs_Started.csv'
name7='Wait_Hours_Per_Job.csv'
name8='Wait_Hours_Total.csv'
name9='Wall_Hours_Per_Job.csv'
name10='Wall_Hours_Total.csv'




data = pd.read_csv(name)
data2= pd.read_csv(name2)
data3= pd.read_csv(name3)
data4= pd.read_csv(name4)
data5= pd.read_csv(name5)
data6= pd.read_csv(name6)
data7= pd.read_csv(name7)
data8= pd.read_csv(name8)
data9= pd.read_csv(name9)
data10= pd.read_csv(name10)
finalData= pd.DataFrame()
lUsers=list(data)

#user='[Golduck] '
for i in range(2,7):
    user=lUsers[i]

#finalData['User']='A'

    #finalData['Day']= data['Day']
    #finalData['User']=user
    finalData[name7]=data7[user]
    finalData[name8]=data8[user]
    finalData[name]=data[user]
    finalData[name2]=data2[user]
    finalData[name3]=data3[user]
    finalData[name4]=data4[user]
    finalData[name5]=data5[user]
    finalData[name6]=data6[user]
    finalData[name9]=data9[user]
    finalData[name10]=data10[user]
    #finalData.index= range(1,len(data)+1)
    #finalData.index.name='Day'
    for k,v in finalData.iterrows():
        if(v[name]==0 and v[name2]==0 and v[name6]==0):
            finalData.at[k,name10]=0
            finalData.at[k,name7]=0
            finalData.at[k,name8]=0
            finalData.at[k,name9]=0
    finalData.to_csv(user+'1')
y.index= range(1,len(data)+1)
y=finalData[name7]

finalData=finalData.drop(columns=[name7])
x=finalData

## DETERMINE SIGNIFICANCE OF A COLUMN
from sklearn import linear_model
lm = linear_model.LinearRegression()
model = lm.fit(x,y)
import statsmodels.api as sm
X=sm.add_constant(x)
model1=sm.OLS(y,X).fit()
model1.summary()

##DROP NON SIGNIFICANT COLUMNS
#finalData=finalData.drop(columns=[name])
finalData=finalData.drop(columns=[name2])
finalData=finalData.drop(columns=[name4])
finalData=finalData.drop(columns=[name5])
#finalData=finalData.drop(columns=[name6])
finalData=finalData.drop(columns=[name9])
finalData=finalData.drop(columns=[name10])

##DETERMINE LINEAR REGRESSION
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
x=finalData
X_train = x[:-20]
X_test = x[-20:]
y_train = y[:-20]
y_test = y[-20:]
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
print(y_pred)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
accuracy = regr.score(X_test,y_test)
print(accuracy*100,'%')

# Plot outputs
plt.scatter(X_test.values[:,1],y_test,  color='black')
plt.plot(X_test.values[:,1], y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

## CREATE A ELASTIC NET MODEL
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
x=finalData
X_train = x[:-20]
X_test = x[-20:]
y_train = y[:-20]
y_test = y[-20:]
regr = ElasticNet(random_state=0)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print(y_pred)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
accuracy = regr.score(X_test,y_test)
print(accuracy*100,'%')

# Plot outputs
plt.scatter(X_test.values[:,1],y_test,  color='black')
plt.plot(X_test.values[:,1], y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


##PLOT CORRELATION
import matplotlib.pyplot as plt
plt.matshow(finalData.corr())
plt.show()