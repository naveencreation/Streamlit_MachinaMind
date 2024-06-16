import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Function to train and save the model
def train_and_save_model(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save the model
    joblib.dump(model, 'model.joblib')
    
    return model, accuracy

# Function to make predictions using the saved model
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit app
def main():
    st.title(":violet[Model Training and Prediction] ")

    # Use session state to manage the state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'accuracy' not in st.session_state:
        st.session_state.accuracy = None
    if 'input_data' not in st.session_state:
        st.session_state.input_data = {}

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(st.session_state.data.head())

        st.session_state.target_column = st.selectbox("Select the target column", st.session_state.data.columns)
        
        if st.button("Train Model"):
            st.session_state.model, st.session_state.accuracy = train_and_save_model(st.session_state.data, st.session_state.target_column)
            st.write(f"Model trained with accuracy: {st.session_state.accuracy:.2f}")

    if st.session_state.model is not None:
        st.write("Model is ready for predictions.")

        for col in st.session_state.data.columns:
            if col != st.session_state.target_column:
                st.session_state.input_data[col] = st.number_input(f"Input value for {col}", value=st.session_state.input_data.get(col, 0), key=f"input_{col}")

        input_df = pd.DataFrame([st.session_state.input_data])
        st.write("Input Data for Prediction:")
        st.write(input_df)

        if st.button("Predict"):
            prediction = predict(st.session_state.model, input_df)
            st.write(f"The predicted value for {st.session_state.target_column} is: {prediction[0]}")

if __name__ == "__main__":
    main()
