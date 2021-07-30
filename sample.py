from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt

#load the 'test.csv' file
df1 = pd.read_csv('area.csv')
#create the linear regression model object
model = linear_model.LinearRegression()
#Train the model with data
model.fit(df1[['Area']], df1.Price)
#To test the model prediction to be done.
output = model.predict([[3400]])
print(output)
#scatter the datapoints to get idea about which ML algorithm can be applied to the dataset.
plt.plot(df1.Area, df1.Price, marker="+", color="red")
#plot the output with linear straight line.
plt.scatter(df1.Area, model.predict(df1[['Area']]), color="blue")
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()


