import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#loading the california house pricing Dataset
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
type(housing)
housing.keys()

#description
print(housing.DESCR)

print(housing.data)
print(housing.feature_names)
print(housing.target)
print(housing.target_names)
print(housing.frame)


#preparing the dataset
df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
df.head()


df['Price'] = housing.target
df.head()

df.info()
df.describe()
print(df.isnull().sum())


#Visualization and analysis of the data
#gives correleation btw the features
#1: Perfect positive correlation (as one variable increases, the other increases proportionally).
#0: No linear correlation (no linear relationship between the variables).
#-1: Perfect negative correlation (as one variable increases, the other decreases proportionally).


df.corr()

plt.figure(figsize=(4,4))
plt.scatter(df['MedInc'], df['Price'], alpha=0.5)
plt.title('Median Income vs. House Value')
plt.xlabel('Median Income')
plt.ylabel('House Value')
plt.show()


import seaborn as sns
#sns.pairplot(df)

plt.figure(figsize=(4, 4))
sns.regplot(x="AveBedrms", y="Price", data=df)

plt.figure(figsize=(4, 4)) 
# Create the regression plot
sns.regplot(x="MedInc", y="Price", data=df)
plt.show()

corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


df.hist(bins=50, figsize=(20,15))
plt.show()

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

x.head
y


#train test split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=32)

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)


x_train

x_test


#pre-processing
#standardizing the dataset
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
x_train
x_test

#Model training
#using Linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

model.coef_

model.intercept_

#info on which parameters model is trained
model.get_params()

# Making predictions
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Calculate residuals
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

# Print residuals
print("Train Residuals:\n", train_residuals)
print("Test Residuals:\n", test_residuals)


#Ploting Residuals for Training Data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, train_residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Training Data: Predicted vs Residuals')


#Ploting Residuals for Test Data
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, test_residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Test Data: Predicted vs Residuals')

plt.tight_layout()
plt.show()

sns.displot(train_residuals, kind="kde", height=4, aspect=1)

sns.displot(test_residuals, kind="kde", height=4, aspect=1)

# Evaluate the model
from sklearn.metrics import mean_squared_error,mean_absolute_error

test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

# Print metrics
print(f'Test Mean Sq Error: {test_mse:.2f}')
print(f'Test Mean Absolute Error: {test_mae:.2f}')



#Code to Calculate and Print R² and Adjusted R²

from sklearn.metrics import r2_score

# Calculate R² score
test_r2 = r2_score(y_test, y_test_pred)

# Calculate Adjusted R² score
n = len(y_test)  # Number of observations
p = x_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)


# Print metrics
print(f'Test R²: {test_r2:.2f}')
print(f'Adjusted R²: {adjusted_r2:.2f}')

#new data prediction
new_data = pd.DataFrame([x.iloc[0].values], columns=x.columns)


# Transform the new data using the fitted scaler
new_data_scaled = scaler.transform(new_data)

# Make predictions with the trained model
model.predict(new_data_scaled)

prediction=model.predict(new_data_scaled)
print('Prediction for new data:\n',prediction)

#pickling the model for deployment
import pickle

# Save the model to a file
with open('regmodel.pkl', 'wb') as file:
    pickle.dump(model, file)


# Load the model from the file
with open('regmodel.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

print("Model loaded successfully.")

loaded_model.predict(new_data_scaled)
