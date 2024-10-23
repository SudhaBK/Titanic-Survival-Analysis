import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score
import pickle

# Load data
train_df = pd.read_csv('Titanic_train.csv')
test_df = pd.read_csv('Titanic_test.csv')

# Preprocess data
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Drop unnecessary columns
train_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline with imputation and classification
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Define hyperparameter tuning space
param_grid = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [None, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Evaluate best model on validation set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)

print("Validation Accuracy (Best Model):", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Save best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Make predictions on test data
test_data = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Embarked'] = test_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

predictions = best_model.predict(test_data)
print("Predictions:")
print(predictions)

# Save predictions to file
pd.DataFrame(predictions).to_csv('predictions.csv', index=False)

# Visualize predictions
sns.countplot(x=predictions)
plt.title = "Survival Predictions"
plt.show()

# Precision and Recall calculation
def calculate_precision_recall(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return precision, recall

# Cross-validation
def perform_cross_validation(model, X, y):
    kf = KFold(n_splits=5, shuffle=True)
    scores = cross_val_score(model, X, y, cv=kf)
    return scores