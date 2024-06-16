import streamlit as st
import streamlit as st
import pandas as pd
from scipy.io import arff
import io
import zipfile
import tempfile
import os
import requests
def main():
    st.write("Hello i am :violet[Data]")
def process_file(file, file_type):
    try:
        if file_type == "csv":
            df = pd.read_csv(file)
        elif file_type == "arff":
            arff_data = file.read()
            data, meta = arff.loadarff(io.StringIO(arff_data.decode('utf-8')))
            df = pd.DataFrame(data)
        elif file_type in ["xls", "xlsx"]:
            df = pd.read_excel(file)
        elif file_type == "json":
            df = pd.read_json(file)
        elif file_type == "txt":
            df = pd.read_csv(file, delimiter='\t')
        else:
            df = None
        return df
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        return None

def main():
    st.title(":violet[Dataset Upload and Download]")
    st.write("You can upload any type of dataset here, whether it's an ARFF file or a Zip file, and it will automatically convert it into a suitable CSV file format for preprocessing")

    option = st.selectbox(
        "Select an option:",
        ("Upload a dataset file", "Download from Kaggle", "Download from UCI ML Repository", "Download from OpenML")
    )

    if option == "Upload a dataset file":
        uploaded_file = st.file_uploader("Upload a dataset file", type=["csv", "arff", "xls", "xlsx", "json", "txt", "zip"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".zip"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                        zip_ref.extractall(tmp_dir)
                        for filename in zip_ref.namelist():
                            file_path = os.path.join(tmp_dir, filename)
                            st.write(f"Processing file: {filename}")
                            file_ext = filename.split('.')[-1].lower()
                            with open(file_path, 'rb') as file:
                                df = process_file(file, file_ext)
                                if df is not None:
                                    st.write(f"Successfully processed {filename}")
                                    st.write(df)

                                    # Convert to CSV and provide download link
                                    csv_file = f"converted_{filename}.csv"
                                    df.to_csv(csv_file, index=False)
                                    with open(csv_file, "rb") as f:
                                        st.download_button(
                                            label=f"Download CSV for {filename}",
                                            data=f,
                                            file_name=csv_file,
                                            mime="text/csv"
                                        )
                            os.remove(file_path)
            else:
                file_ext = uploaded_file.name.split('.')[-1].lower()
                df = process_file(uploaded_file, file_ext)
                if df is not None:
                    st.write(df)

                    # Convert to CSV and provide download link
                    csv_file = "converted_dataset.csv"
                    df.to_csv(csv_file, index=False)
                    with open(csv_file, "rb") as f:
                        st.download_button(
                            label="Download CSV",
                            data=f,
                            file_name=csv_file,
                            mime="text/csv"
                        )
    elif option == "Download from Kaggle":
        st.markdown("[Go to Kaggle](https://www.kaggle.com/datasets)")

    elif option == "Download from UCI ML Repository":
        st.markdown("[Go to UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)")

    elif option == "Download from OpenML":
        st.markdown("[Go to OpenML](https://www.openml.org)")

if __name__ == "__main__":
    main()
