import streamlit as st
from streamlit_lottie import st_lottie
import json

def main():
    st.title(f":violet[Welcome to My Home Page], {st.session_state['username']}!")
    st.write("Before diving into file preprocessing or any other procedures, it's crucial to establish a solid foundation in machine learning principles.")
    #st.title("Machine Learning")
    st.subheader(":blue[The Machine Learning Life Cycle]")
    st.image("beginner.png", use_column_width=True)
    st.write("Lets Learn about the steps involved in a standard machine learning project as we explore the ins and outs of the machine learning lifecycle using CRISP-ML(Q).")

    #st.write("MachineLearningLifecycle")
    st.write("""
    Let's explore the steps of a typical machine learning project using CRISP-ML(Q), a widely accepted methodology in the industry.

Machine learning projects involve much more than just data processing, training models, and deployment. They encompass business understanding, data collection, analytics, model building, evaluation, deployment, and ongoing monitoring and maintenance.

The machine learning lifecycle provides a structured approach to projects, efficiently allocating resources and ensuring the development of sustainable, cost-effective AI products.
    """)

    st.subheader(":blue[The 6 steps in a standard machine learning life cycle:]")
    st.image("lifecycle.png", caption="Your Image", use_column_width=True)
    st.write("""
    1. Planning
    2. Data Preparation 
    3. Model Engineering
    4. Model Evaluation
    5. Model Deployment
    6. Monitoring and Maintenance
    """)

    st.write("""
    Each phase in the machine learning cycle follows a quality assurance framework for constant improvement and maintenance by strictly following requirements and constraints. Learn more about quality assurance by reading the CRISP-ML(Q) blog. 

    For non-technical individuals and managers, check out our short course on Understanding Machine Learning fundamentals. It will help them understand machine learning in general, modeling, and deep learning (AI). You can also explore the differences between AI and machine learning in a separate article. 
    """)

    st.subheader(":blue[1. Planning]")
    st.write("Machine Learning Project Planning")
    st.image("planning.png", use_column_width=True)
    st.write("""
    The planning phase involves assessing the scope, success metric, and feasibility of the ML application. You need to understand the business and how to use machine learning to improve the current process. For example: do we require machine learning? Can we achieve similar requests with simple programming?

    You also need to understand the cost-benefit analysis and how you will ship the solution in multiple phases. Furthermore, you need to define clear and measurable success metrics for business, machine learning models (Accuracy, F1 score, AUC), and economic (key performance indicators).

    Finally, you need to create a feasibility report. 
    """)

    st.subheader(":blue[2. Data Preparation]")
    st.image("data_extraction.png", width=300)
    st.write("""
    The data preparation section is further divided into four parts: data procurement and labeling, cleaning, management, and processing.   

    Data collection and labeling
    We need first to decide how we will collect the data by gathering the internal data, open-source, buying it from the vendors, or generating synthetic data. Each method has pros and cons, and in some cases, we get the data from all four methodologies. 

    After collection, we need to label the data. Buying cleaned and labeled data is not feasible for all companies, and you may also need to make changes to the data selection during the development process. That is why you cannot buy it in bulk and why the data can eventually be useless for the solution. 
    """)

    st.subheader(":blue[3. Model Engineering]")
    st.image("feature-engineering.png", use_column_width=True)
    st.write("""
    In this phase, we will be using all the information from the planning phase to build and train a machine learning model. For example: tracking model metrics, ensuring scalability and robustness, and optimizing storage and compute resources. 

    Build effective model architecture by doing extensive research.
    Defining model metrics.
    Training and validating the model on the training and validation dataset. 
    Tracking experiments, metadata, features, code changes, and machine learning pipelines.
    Performing model compression and ensembling. 
    Interpreting the results by incorporating domain knowledge experts. 
    We will be focusing on model architecture, code quality, machine learning experiments, model training, and ensembling. 

    The features, hyperparameters, ML experiments, model architecture, development environment, and metadata are stored and versioned for reproducibility. 

    Learn about the steps involved in model engineering by taking the Machine Learning Scientist with Python career track. It will help you master the necessary skills to land a job as a machine learning engineer.
    """)

    st.subheader(":blue[4. Model Evaluation]")
    st.image("Model_Evaluation.png", use_column_width=True)
    st.write("""
    Now that we have finalized the version of the model, it is time to test various metrics. Why? So that we can ensure that our model is ready for production. 

    We will first test our model on a test dataset and make sure we involve subject matter experts to identify the error in the predictions. 

    We also need to ensure that we follow industrial, ethical, and legal frameworks for building AI solutions. 

    Furthermore, we will test our model for robustness on random and real-world data. Making sure that the model inferences fast enough to bring the value. 

    Finally, we will compare the results with the planned success metrics and decide on whether to deploy the model or not. In this phase, every process is recorded and versioned to maintain quality and reproducibility. 
    """)

    st.subheader(":blue[5. Model Deployment]")
    st.image("Model_Deployment.png", use_column_width=True)
    st.write("""
    In this phase, we deploy machine learning models to the current system. For example: introducing automatic warehouse labeling using the shape of the product. We will be deploying a computer vision model into the current system, which will use the images from the camera to print the labels.

    Generally, the models can be deployed on the cloud and local server, web browser, package as software, and edge device. After that, you can use API, web app, plugins, or dashboard to access the predictions. 

    In the deployment process, we define the inference hardware. We need to make sure we have enough RAM, storage, and computing power to produce fast results. After that, we will evaluate the model performance in production using A/B testing, ensuring user acceptability. 

    The deployment strategy is important. You need to make sure that the changes are seamless and that they have improved the user experience. Moreover, a project manager should prepare a disaster management plan. It should include a fallback strategy, constant monitoring, anomaly detection, and minimizing losses. 
    """)

    st.subheader(":blue[6. Monitoring and Maintenance]")
    st.image("D:/Software developnment/Try/image/Monitoring_and_Maintenance.png", use_column_width=True)
   
