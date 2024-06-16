import streamlit as st
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import pickle
import tempfile

def initialize_h2o():
    h2o.init()

def shutdown_h2o():
    h2o.shutdown()

def train_model(df):
    h2o_df = h2o.H2OFrame(df)
    x = h2o_df.columns[:-1]
    y = h2o_df.columns[-1]
    h2o_df[y] = h2o_df[y].asfactor()
    
    aml = H2OAutoML(max_models=20, seed=1)
    aml.train(x=x, y=y, training_frame=h2o_df)
    
    return aml

def main():
    st.title(':violet[H2O AutoML with Streamlit]')

    st.write("This app trains an H2O AutoML model and allows you to download the trained model as a pickle file.")
    
    initialize_h2o()

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(data.head())
        
        if st.button('Train Model'):
            st.write("Training model...")
            model = train_model(data)
            st.write("Model training completed.")
            
            # Save the model as a pickle file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                pickle.dump(model, tmp_file)
                tmp_file.seek(0)
                model_pickle = tmp_file.read()
                
            st.download_button(
                label="Download Model as Pickle",
                data=model_pickle,
                file_name='h2o_automl_model.pkl',
                mime='application/octet-stream'
            )

    if st.button('Stop H2O'):
        shutdown_h2o()

if __name__ == "__main__":
    main()
