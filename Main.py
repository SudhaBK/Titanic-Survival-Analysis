import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Define numeric and categorical features
numeric_features = ['Age', 'SibSp', 'Parch']
categorical_features = ['Sex', 'Pclass']

# Define preprocessors
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=300, random_state=42))
])

# Load data
train_df = pd.read_csv('Titanic_train.csv')

# Train model
X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(train_df.drop('Survived', axis=1))

# Streamlit app
st.write("Titanic Survival Prediction")
st.write("-------------------------------")

# Input form
with st.form("input_form"):
    Pclass = st.selectbox("Passenger Class", [1, 2, 3])
    Sex = st.selectbox("Sex", ["male", "female"])
    Age = st.number_input("Age", min_value=1, max_value=120, value=21, step=1)
    SibSp = st.number_input("Siblings/Spouses", min_value=0, max_value=10, value=0, step=1)
    Parch = st.number_input("Parents/Children", min_value=0, max_value=10, value=0, step=1)
    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    input_df = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])
    prediction = pipeline.predict(input_df)
    if prediction[0] == 1:
        st.write("Survival Prediction: Survived")
    else:
        st.write("Survival Prediction: Not Survived")

# Display predictions
st.write("Training Predictions:")
survived_counts = y_pred.sum()
not_survived_counts = len(y_pred) - survived_counts
st.write(f"Survived: {survived_counts} ({survived_counts/len(y_pred)*100:.2f}%)")
st.write(f"Not Survived: {not_survived_counts} ({not_survived_counts/len(y_pred)*100:.2f}%)")