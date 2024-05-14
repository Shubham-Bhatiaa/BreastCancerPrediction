import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Function to load data
def load_data():
    breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
    data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
    data_frame['label'] = breast_cancer_dataset.target
    return data_frame

# Function to train the model
def train_model(data):
    X = data.drop(columns='label', axis=1)
    Y = data['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    return model, X_train, X_test, Y_train, Y_test

# Function to predict
def predict(model, input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    return prediction

# Main function
def main():
    st.title("Breast Cancer Prediction")
    st.write("This application predicts whether a breast cancer tumor is malignant (1) or benign (0).")

    # Load data
    data = load_data()

    # Train the model
    model, X_train, X_test, Y_train, Y_test = train_model(data)

    # Display accuracy
    training_data_accuracy = accuracy_score(Y_train, model.predict(X_train))
    test_data_accuracy = accuracy_score(Y_test, model.predict(X_test))
    st.write(f"Training Data Accuracy: {training_data_accuracy}")
    st.write(f"Test Data Accuracy: {test_data_accuracy}")

    # Input form
    st.sidebar.header('Input Parameters')
    input_data = []
    for feature in data.columns[:-1]:
        value = st.sidebar.slider(f"Enter {feature}", float(data[feature].min()), float(data[feature].max()))
        input_data.append(value)

    if st.sidebar.button("Predict"):
        prediction = predict(model, input_data)
        if prediction[0] == 0:
            st.write('The Breast cancer is benign.')
        else:
            st.write('The Breast cancer is malignant.')

if __name__ == "__main__":
    main()
