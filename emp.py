import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Set page layout
st.set_page_config(page_title="HR Attrition Prediction", page_icon=":bar_chart:", layout="centered")

# Set background color and page width
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #ffecd2, #fcb69f);  /* fallback for old browsers */
        background: linear-gradient(to right, #ffecd2, #fcb69f);
        max-width: 85%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the dataset
df = pd.read_csv('HR-Employee-Attrition.csv')

# Drop unnecessary columns
columns_to_drop = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
df.drop(columns=columns_to_drop, inplace=True)

# Encoding the target variable
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])

# Display the dataset
st.write("## HR Employee Attrition Dataset")
st.dataframe(df.head())

# Exploratory Data Analysis
st.write("## Exploratory Data Analysis")

# Distribution of the target variable
st.write("### Distribution of Attrition")
st.bar_chart(df['Attrition'].value_counts())

# Pairplot with some selected features
selected_features = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'Attrition']
pairplot = sns.pairplot(df[selected_features], hue='Attrition')
st.pyplot(pairplot)

# Model Building
st.write("## Model Building")

# Perform one-hot encoding on categorical variables
df_encoded = pd.get_dummies(df)

# Splitting the dataset into features (X) and target (y)
X = df.drop(['Attrition'], axis=1)
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

st.write("### Model Evaluation")
st.write(f"Accuracy: {accuracy}")
st.write(f"ROC AUC Score: {roc_auc}")
st.write("Confusion Matrix:")
st.write(conf_matrix)
st.write("Classification Report:")
st.write(class_report)

# Most risky employees
st.write("### Most Risky Employees")
probs = rf.predict_proba(X_test)[:, 1]
probs_df = pd.DataFrame({'Probability': probs, 'Employee ID': X_test.index})
top_risky = probs_df.sort_values(by='Probability', ascending=False).head(10)
st.dataframe(top_risky)
