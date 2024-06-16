import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import base64

# Function to replace specified symbols with 0
def replace_symbols_with_zero(df):
    symbols = ['/', '?', '@', '#', '$', '%', '&', '*', '!', '^', '(', ')', '-', '+', '=', '[', ']', '{', '}', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.']
    for symbol in symbols:
        df.replace(symbol, 0, inplace=True)
    return df

# Function to handle missing values based on user selection
def preprocess_data(df, num_method, cat_method, num_fill_value=None, cat_fill_value=None, num_strategy='mean', cat_strategy='most_frequent'):
    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(exclude='number').columns
    
    # Handle numeric columns
    if num_method == 'drop':
        df = df.dropna(subset=numeric_cols)
    elif num_method == 'fill':
        df[numeric_cols] = df[numeric_cols].fillna(num_fill_value)
    elif num_method == 'impute':
        num_imputer = SimpleImputer(strategy=num_strategy)
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    
    # Handle categorical columns
    if cat_method == 'drop':
        df = df.dropna(subset=categorical_cols)
    elif cat_method == 'fill':
        df[categorical_cols] = df[categorical_cols].fillna(cat_fill_value)
    elif cat_method == 'impute':
        cat_imputer = SimpleImputer(strategy=cat_strategy, fill_value=cat_fill_value)
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    return df

# Function to encode categorical data based on user selection
def encode_data(df, encoding_method, selected_cols=None):
    categorical_cols = df.select_dtypes(exclude='number').columns
    if encoding_method == 'Label Encoding':
        if selected_cols:
            for col in selected_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        else:
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
    elif encoding_method == 'One-Hot Encoding':
        if selected_cols:
            df = pd.get_dummies(df, columns=selected_cols, drop_first=True)
        else:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

# Function for the main preprocessing app
def preprocessing_app():
    st.title(":violet[File Preprocessing]")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        # Clear session state when new file is uploaded
        st.session_state.pop('columns_to_drop', None)

        # Read CSV file
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview")
        st.dataframe(df.head())

        # Replace symbols with 0
        df = replace_symbols_with_zero(df)

        # Display missing values
        st.write("Missing Values Summary:")
        st.write(df.isnull().sum())

        # Preprocessing options for numeric columns
        num_method = st.selectbox("Choose method to handle missing values for numeric columns", ["drop", "fill", "impute"])
        
        num_fill_value = None
        num_strategy = 'mean'
        
        if num_method == 'fill':
            num_fill_value = st.number_input("Enter value to fill missing numeric data", value=0)
        elif num_method == 'impute':
            num_strategy = st.selectbox("Choose imputation strategy for numeric data", ["mean", "median", "most_frequent", "constant"])
            if num_strategy == 'constant':
                num_fill_value = st.number_input("Enter constant value to fill missing numeric data", value=0)

        # Preprocessing options for categorical columns
        cat_method = st.selectbox("Choose method to handle missing values for categorical columns", ["drop", "fill", "impute"])
        
        cat_fill_value = None
        cat_strategy = 'most_frequent'
        
        if cat_method == 'fill':
            cat_fill_value = st.text_input("Enter value to fill missing categorical data")
        elif cat_method == 'impute':
            cat_strategy = st.selectbox("Choose imputation strategy for categorical data", ["most_frequent", "constant"])
            if cat_strategy == 'constant':
                cat_fill_value = st.text_input("Enter constant value to fill missing categorical data")

        # Encoding options for categorical columns
        encoding_method = st.selectbox("Choose Encoding method for categorical columns", ["None", "Label Encoding", "One-Hot Encoding"])

        # Option for selecting specific categorical columns for encoding
        selected_categorical_cols = st.multiselect("Select categorical columns for encoding", df.select_dtypes(exclude='number').columns)

        # Store selected columns to drop in session state
        if 'columns_to_drop' not in st.session_state:
            st.session_state['columns_to_drop'] = []

        if st.button("Add Column to Drop"):
            selected_column = st.selectbox("Select column to drop", df.columns)
            st.session_state['columns_to_drop'].append(selected_column)

        # Display selected columns to drop
        if st.session_state['columns_to_drop']:
            st.write("Columns to Drop:")
            st.write(st.session_state['columns_to_drop'])

        if st.button("Preprocess Data"):
            # Drop selected columns
            preprocessed_df = df.drop(columns=st.session_state['columns_to_drop'])

            # Preprocess data
            preprocessed_df = preprocess_data(preprocessed_df, num_method, cat_method, num_fill_value, cat_fill_value, num_strategy, cat_strategy)

            # Encode data
            if encoding_method != "None":
                preprocessed_df = encode_data(preprocessed_df, encoding_method, selected_categorical_cols)

            # Remove 'Unnamed' columns
            preprocessed_df = preprocessed_df.loc[:, ~preprocessed_df.columns.str.contains('^Unnamed')]
            
            st.write("Preprocessed Data Preview:")
            st.dataframe(preprocessed_df.head())

            # Download link for preprocessed data
            csv = preprocessed_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
            href = f'<a href="data:file/csv;base64,{b64}" download="preprocessed_data.csv">Download Preprocessed CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)

# Main function to control page navigation
def main():
    preprocessing_app()

if __name__ == "__main__":
    main()
