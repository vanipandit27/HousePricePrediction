# HousePricePrediction
 • Developed a robust real estate price prediction system using a regression algorithm to analyze multiple characteristics such as square footage, number of rooms,  location, and amenities, achieving an impressive average prediction. 
 • Optimized Regression Model Performance Through Strategic Feature Engineering.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
%matplotlib inline

data=pd.read_csv("/content/kc_house_data.csv")

data.head()

data.describe()

data.isnull()

data.isnull().sum().sum()

data.dropna(inplace=True)

data['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine

plt.figure(figsize=(10,10))
sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()
sns.despine

plt.scatter(data.price, data.sqft_living)
plt.title("Price vs Square Feet")

plt.scatter(data.price, data.long)
plt.title("Price vs Location of the area")

plt.scatter(data.price, data.lat)
plt.title("Latitude vs Price")

plt.scatter(data.bedrooms,data.price)
plt.title("Bedroom and Price")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine

plt.scatter((data['sqft_living']+data['sqft_basement']),data['price'])

plt.scatter(data.waterfront,data.price)
plt.title("Waterfront vs Price (0=no waterfront)")

plt.scatter(data.floors,data.price)

plt.scatter(data.condition,data.price)

plt.scatter(data.zipcode,data.price)
plt.title("Which is the pricey location by zipcode?")

from sklearn.linear_model import LinearRegression

reg=LinearRegression()

labels=data['price']
conv_dates=[1 if values==2014 else 0 for values in data.date]
data['date']=conv_dates
train1=data.drop(['id', 'price'],axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(train1, labels, test_size=0.10, random_state=2)

reg.fit(x_train,y_train)

reg.score(x_test,y_test)
