import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# -------------------------------
# Load and preprocess dataset
# -------------------------------
@st.cache_resource
def load_data():
    df = pd.read_csv("Chronic_Kidney_Disease.csv")
    df.replace('?', np.nan, inplace=True)

    # Convert numerical columns
    num_cols = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(df.mean(), inplace=True)

    cat_cols = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','class']
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop('class', axis=1)
    y = df['class']

    return X, y

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Train Models
# -------------------------------
@st.cache_resource
def train_models():
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB()
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

models = train_models()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🩺 Chronic Kidney Disease Prediction")
st.write("Enter patient details and predict CKD using different ML models.")

# Collect user input
def user_input():
    data = {}
    for col in X.columns:
        data[col] = st.number_input(f"Enter {col}", value=0.0)
    features = pd.DataFrame([data])
    return features

input_data = user_input()

# Preprocess input
input_scaled = scaler.transform(input_data)

# Choose Model
model_choice = st.selectbox("Choose Model", list(models.keys()))

# Predict
if st.button("Predict"):
    model = models[model_choice]
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None

    result = "CKD Detected" if prediction == 1 else "No CKD"
    st.subheader(f"Prediction using {model_choice}: {result}")

    if prob is not None:
        st.write(f"Probability of CKD: {prob*100:.2f}%")
