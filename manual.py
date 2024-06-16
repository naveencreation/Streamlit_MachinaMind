import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score,
                             mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error)
import seaborn as sns
import matplotlib.pyplot as plt

# Function to perform data analysis
def analyze_data(df):
    st.subheader(":blue[Data Overview]")
    st.write(df.head())

    st.subheader(":blue[Summary Statistics]")
    st.write(df.describe(include='all'))

    st.subheader(":blue[Missing Values]")
    st.write(df.isnull().sum())

    st.subheader(":blue[Data Distribution]")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Create rows of plots with two plots per row
    num_cols = 2
    for i in range(0, len(numeric_columns), num_cols):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            if i + j < len(numeric_columns):
                col_name = numeric_columns[i + j]
                with cols[j]:
                    st.write(f"Distribution of {col_name}")
                    fig = px.histogram(df, x=col_name, nbins=30, marginal="box", title=f"Distribution of {col_name}")
                    fig.update_layout(width=450, height=350)
                    st.plotly_chart(fig, use_container_width=True)

    st.subheader(":blue[Correlation Matrix]")
    corr = df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.index.values,
        y=corr.columns.values,
        colorscale='Viridis',
        text=corr.values,
        texttemplate="%{text:.2f}"
    ))
    fig.update_layout(title='Correlation Matrix', width=900, height=600)
    st.plotly_chart(fig, use_container_width=True)

    return numeric_columns, df.select_dtypes(include=['object']).columns

# Function to encode categorical features
def encode_categorical_features(df, categorical_columns):
    encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col].astype(str))
    return df

# Function to evaluate classification models
def evaluate_classification_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Support Vector Machine': SVC(probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    results = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if len(set(y_test)) == 2 else None

        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        })

    return pd.DataFrame(results)

# Function to evaluate regression models
def evaluate_regression_models(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Support Vector Machine': SVR(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor()
    }

    results = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        results.append({
            'Model': model_name,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R-squared': r2,
            'MAPE': mape
        })

    return pd.DataFrame(results)

# Main function for Streamlit app
def main():
    st.title(":violet[File Analysis and Model Selection]")

    uploaded_file = st.file_uploader("Upload your dataset (.csv file)", type="csv")
    if uploaded_file is not None:
        st.write("### :blue[Dataset Preview:]")
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        st.title(":blue[Data Analysis:]")
        numeric_columns, categorical_columns = analyze_data(df)

        st.write("###  :blue[Select Target Column:]")
        target_column = st.selectbox("Select the target column", options=df.columns)

        if df.empty or target_column is None:
            st.warning("Please upload a valid dataset and select a target column.")
            return

        st.write("###  :blue[Select Independent Feature Columns:]")
        independent_columns = st.multiselect("Select the independent feature columns", options=df.columns.tolist())

        if not independent_columns:
            st.warning(" :Please select at least one independent feature column.")
            return

        X = df[independent_columns]
        y = df[target_column]

        task_type = st.radio("Select Task Type:", ("Classification", "Regression"))

        # Check if there are enough samples for train-test split
        if len(df) < 2:
            st.warning("Dataset is too small to split into training and testing sets.")
            return

        # Train-test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except ValueError as e:
            st.error(f"Error occurred during train-test split: {str(e)}")
            return

        if task_type == "Classification":
            st.write("### :blue[Classification Models Evaluation:]")
            results_classification = evaluate_classification_models(X_train, X_test, y_train, y_test)
            st.write(results_classification)

            if st.button("Generate Classification Heatmap"):
                st.write("Generating Classification Heatmap...")
                st.write("Please wait, this may take a moment depending on the size of your dataset.")

                # Calculate the correlation matrix
                corr = df.corr()

                # Create a heatmap
                fig, ax = plt.subplots(figsize=(12, 10))  # Adjust the size of the heatmap
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})  # Adjust font size with annot_kws
                st.pyplot(fig)

        elif task_type == "Regression":
            st.write("### Regression Models Evaluation:")
            results_regression = evaluate_regression_models(X_train, X_test, y_train, y_test)
            st.write(results_regression)

            if st.button("Generate Regression Heatmap"):
                st.write("Generating Regression Heatmap...")
                st.write("Please wait, this may take a moment depending on the size of your dataset.")

                # Calculate the correlation matrix
                corr = df.corr()

                # Create a heatmap
                fig, ax = plt.subplots(figsize=(12, 10))  # Adjust the size of the heatmap
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})  # Adjust font size with annot_kws
                st.pyplot(fig)

if __name__ == "__main__":
    main()
