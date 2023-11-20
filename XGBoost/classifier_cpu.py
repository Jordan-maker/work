import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys

# Loading the dataset (~580K rows and 54 columns for X)
X, y = datasets.fetch_covtype(return_X_y=True)
y -= y.min()

classes = np.unique(y)

# Split the training sample (70%) from test sample (30%).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create an XGBoost classifier
#clf = xgb.XGBClassifier(objective='multi:softmax',
#                        num_class=len(classes),
#                        random_state=42,
#                        device='cpu',  # used by default
#                        n_jobs=-1,
#                        )

# Create an XGBoost classifier
clf = xgb.XGBClassifier(objective='multi:softmax',
                        num_class=len(classes),
                        random_state=42,
                        device='cpu',  # used by default
                        n_jobs=-1,
                        )

param_grid = {'n_estimators': [50, 100, 150, 200],
              'learning_rate': [0.02, 0.04, 0.06, 0.08, 0.1],
              'max_depth': [4, 6, 8, 10],
              }

# GridSearch for hyperparam tunning and Cross Validation
search = GridSearchCV(estimator=clf,
                      #param_grid=param_grid,
                      scoring="accuracy",
                      cv=10,        # number of crossVal folds.
                      n_jobs=-1,    # Using all cores for processing
			    )

start=time.time()
# Fit the model evaluating the grid of params
clf.fit(X_train, y_train)
final=time.time()
print(final-start)


# Picking the best estimator
clf_best = search.best_estimator_

# The cross validation is applied on the training sample.
cv_score = cross_val_score(estimator=clf_best,
                           X=X_train, y=y_train.values.ravel(), cv=10, n_jobs=-1)

cv_score_mean = round( cv_score.mean()*100, 2 )
cv_score_std  = round( cv_score.std()*100, 2 )

print(f'({cv_score_mean} +- {cv_score_std})%')

y_pred = clf_best.predict(X_test)

y_ = y_test.copy()
y_['target_pred'] = y_pred


accuracy_on_test = accuracy_score(y_true=y_test, y_pred=y_pred)  # accuracy on the test sample
print(f'accuracy score on the test sample: {round(accuracy_on_test, 4)*100}%')

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted', fontweight='bold')
plt.ylabel('Actual', fontweight='bold')
plt.title('Confusion Matrix for Iris Dataset using XGBoost')
plt.show()




