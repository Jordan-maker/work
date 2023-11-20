import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# For this file, we focus on Scaling data and perform Polynomial Regression

# load Dataframe
df = pd.read_csv('data.csv')

# Now, we can split the Dataframe for training and test
y = df[['Height']]
X = df.drop(y, axis=1)

# Define scalers
# Here, this scaler is valid due to no outliers are presented.
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Applying respective scalers to data
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split the scaled data in training and test samples
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Create a 2-degree polynomial without bias (no demand zero interception)
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_X_train = poly.fit_transform(X_train)
poly_X_test  = poly.fit_transform(X_test)

# Fit the model
model = LinearRegression()
model.fit(poly_X_train, y_train)

# Calculate score of the fit
score = model.score(poly_X_train, y_train)

# Display model coefficients
# The model is described by the equation: y = beta_0 + beta_1*x^1 + beta_2*x^2
beta_0 = model.intercept_[0]
beta_1 = model.coef_[0][0]
beta_2 = model.coef_[0][1]

# Function that describes the model
function = lambda x: beta_0 + beta_1*x + beta_2*x**2

# Generate dense points for the model curve
h_values = np.arange(X_train.min(), X_train.max(), 0.01)   # horizontal (x-axis)
v_values = [function(float(i)) for i in h_values]          # vertical   (y-axis)
v_values = np.array(v_values).reshape(-1, 1)

# Applying inverse to the dense points of model
h_values = scaler_X.inverse_transform(h_values.reshape(-1, 1))
v_values = scaler_y.inverse_transform(v_values)

# Applying inverse to points of the training data
X_train = scaler_X.inverse_transform(X_train)
y_train = scaler_y.inverse_transform(y_train)

# Applying inverse to points of the test data
X_test  = scaler_X.inverse_transform(X_test)
y_test  = scaler_y.inverse_transform(y_test)


# Plot
plt.plot(h_values, v_values,  color='tab:red',  label='Model')
plt.scatter(X_train, y_train, color='tab:blue', label='Training')
plt.scatter(X_test, y_test,   color='tab:green',label='Test')

plt.title("Free Fall")
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")

plt.legend()
plt.show()


# For predict the height for new values of time
def predicting(time:float=1.0):

    df_ = pd.DataFrame([time], columns=['Time'])
    time_scaled = scaler_X.transform(df_)[0][0]
    height_value = function(float(time_scaled))
    height_value = scaler_y.inverse_transform(np.array(height_value).reshape(-1, 1))
    return height_value[0][0]


height_predicted = predicting(time=3.5)
print(height_predicted)











