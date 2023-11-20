import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# load Dataframe
df = pd.read_csv('Salary_Data.csv')

# Spliting features and target
X = df.drop(columns=['Salary'], axis=1)  # Dataframe type
y = df[['Salary']]                       # Dataframe type

# Spliting training and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining scalers
scalerX = StandardScaler()
scalery = StandardScaler()  # Is not needed to scale the target values, even when they have high values

# Scale the training samples before fit
X_train_scaled = scalerX.fit_transform(X_train)   # The first time is used fit_transform(), but later, only transform()
y_train_scaled = scalery.fit_transform(y_train)   # this is not needed

# Performing fitting
model = LinearRegression(n_jobs=-1)
model.fit(X_train_scaled, y_train_scaled)


def predicting(X):
    """
    :param X: Feature dataframe
    :return: predicted target as original-format
    """
    X_scaled = scalerX.transform(X)

    pred_scaled = model.predict(X_scaled)
    pred = scalery.inverse_transform(pred_scaled)

    return pred.astype(int)


def calculate_slope_and_intercept(X, y):
    """
    :param X: X_training dataset
    :param y: y_training target
    :return: List of slopes
    """
    slopes = []
    intercepts = []
    for i in range(len(X.columns)):

        x0 = X.iloc[0, i]   ; x1 = X.iloc[1, i]
        y0 = y.pred.iloc[0] ; y1 = y.pred.iloc[1]

        delta_y = y1-y0
        delta_X = x1-x0

        slope = round(delta_y/delta_X, 2)
        intercept = round(y0-slope*x0, 2)

        slopes.append(slope)
        intercepts.append(intercept)

    return slopes, intercepts

# Adding the prediction values
y_train['pred'] =  predicting(X_train)
y_test['pred']  =  predicting(X_test)

# Calculating Slopes and intercepts in the projections
slopes, intercepts = calculate_slope_and_intercept(X_train, y_train)


get_x_values = lambda feature, n=100: np.linspace(feature.min(), feature.max(), n)
get_y_values = lambda slope, intercept, x_values: slope*x_values + intercept

x_values_feature_0 = get_x_values(X_train.iloc[:, 0])
x_values_feature_1 = get_x_values(X_train.iloc[:, 1])

y_values_feature_0 = get_y_values(slope=slopes[0], intercept=intercepts[0], x_values=x_values_feature_0)
y_values_feature_1 = get_y_values(slope=slopes[1], intercept=intercepts[1], x_values=x_values_feature_1)


# Plot 3D scatter
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax  = fig.add_subplot(projection='3d')

data = {"YearsExperience": x_values_feature_0,
        "Age": x_values_feature_1}

df_to_pred = pd.DataFrame(data)
z_values = predicting(df_to_pred)

line_x = [x_values_feature_0.min(), x_values_feature_0.max()]
line_y = [x_values_feature_1.min(), x_values_feature_1.max()]
line_z = [z_values.min(), z_values.max()]

ax.plot(line_x, line_y, line_z, c='tab:red', label='model')
ax.scatter(X_train.YearsExperience, X_train.Age, y_train.Salary, c='tab:blue',  label='training')
ax.scatter(X_test.YearsExperience,  X_test.Age,  y_test.Salary,  c='tab:green', label='test')

# Set labels
ax.set_xlabel('YearsExperience')
ax.set_ylabel('Age')
ax.set_zlabel('Salary')

plt.legend()
plt.show()








