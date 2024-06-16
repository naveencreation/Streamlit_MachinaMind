import streamlit as st

# Set the page configuration
#st.set_page_config(page_title="Naveen's Portfolio", page_icon=":wave:", layout="wide")

def main():
    # Introduction Section with image in right column
    st.title(":blue[Hello, I am] :violet[Naveen] :wave:")
    
    col1, col2 = st.columns([3, 1])  # Create two columns, with the first column three times wider than the second
    
    with col1:
        st.write("Welcome to my portfolio . Here you can find more about me, my skills, education, and projects.")
        st.write("Thank you for using my app. It uses advanced machine learning to analyze data and make predictions. Simply upload your data, choose a model, configure the settings, and run the analysis. You can also download the results.")
        st.write("If you find any issues or have suggestions, please contact me, and I'll fix them quickly. I hope you enjoy using the app and find it helpful!")
    with col2:
        st.image("Profile.jpg", caption="Naveen.S", width=150)  # Display the image in passport size (approx 150px width)

    # Contact Information Section
    st.header(":blue[Contact Information]")
    st.write("Feel free to reach out to me through the following platforms:")

    # Custom HTML and CSS for icons
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <div style="display: flex; gap: 10px;">
        <a href="https://github.com/naveencreation" target="_blank">
            <i class="fa fa-github" style="font-size:24px"></i>
        </a>
        <a href="https://www.linkedin.com/in/naveen-s-selvan-3b198b269" target="_blank">
            <i class="fa fa-linkedin" style="font-size:24px"></i>
        </a>
        <a href="mailto:naveenselvan0004@gmail.com">
            <i class="fa fa-envelope" style="font-size:24px"></i>
        </a>
    </div>
    """, unsafe_allow_html=True)

    # Educational Qualifications Section
    st.header(":blue[Educational Qualifications]")

    st.write("""
    - **B.tech in Artificial Intelligence and Data Science**
    - Karpagam College of Engineering, Coimbatore
    - CGPA: 8.7
    """)

    # Experience Section
    st.header(":blue[Experience]")

    st.write("""
    **Code Clause** (2022–2026)  
    Led the development and implementation of the model  

    **September 2023 – October 2023**  
    - Worked with Python using Google Colab with Machine Learning.
    - Natural Language Processing (NLP)
    - Machine Learning model training
    - Text data preprocessing using Python.
    """)

    # Certifications Section
    st.header(":blue[Certifications]")

    st.write("""
    **Email Spam Classifier**  
    Code Clause | Google Colab  
    September 2023 – October 2023  
    - Learning Modules: Text preprocessing, NLP techniques, machine learning algorithms, model evaluation.
    - Learning Outcomes: Developed skills in creating and evaluating machine learning models for email classification using Python.

    **Leadership and Team Effectiveness**  
    NPTEL | Online | Completed.  
    January 2023 – April 2023  
    - To develop the interpersonal process & understand about Teamwork.
    - Global leadership skills
    """)

    # Projects Section
    st.header(":blue[Projects]")

    st.write("""
    **:blue[Project 1:] :violet[Email Spam Classifier]**  
    Python, NLP, Scikit-learn, Google Colab  
    September 2023 – October 2023  
    - Develop a machine learning model to classify emails as spam or not spam using natural language processing techniques.
    - Led the development and implementation of the model; utilized Python, Scikit-learn for model training, and Google Colab for coding and collaboration.
    - Successfully created a robust spam classifier with high accuracy, improving email filtering efficiency and reducing spam-related issues.

    **:blue[Project 2:] :violet[Mask Detection]**  
    Python, OpenCV, TensorFlow, Google Colab  
    September 2023 – October 2023  
    - Implement a computer vision model to detect whether individuals in images are wearing masks.
    - Designed and trained a deep learning model using TensorFlow; performed image preprocessing with OpenCV and used Google Colab for model training.
    - Achieved accurate real-time mask detection, contributing to public health monitoring and safety measures during the COVID-19 pandemic.

    **:blue[Project 3:] :violet[Blindness Detection]**  
    Python, TensorFlow, Keras, Google Colab  
    September 2023 – October 2023  
    - Develop a convolutional neural network (CNN) to detect blindness from retinal images.
    - Built and validated the CNN model; used Python with TensorFlow and Keras for deep learning, and Google Colab for execution.
    - Enhanced diagnostic capabilities by providing a reliable tool for early blindness detection, aiding in timely medical intervention.

    **:blue[Project 4:] :violet[GitHub Clone]**  
    HTML, CSS, JavaScript, Node.js, MongoDB, Python  
    March 2024 – April 2024  
    - Clone the functionality of GitHub and improve some of the options like pull, push methods.
    - Created the interactive user interface and connect the both frontend and backend using Python.
    - Simplified the use of commands and helped to gain more knowledge about full-stack development.
    """)

    # Skills Section
    st.header(":blue[Skills]")
    st.write("""
    - **Programming Languages**: Python, JavaScript, C++
    - **Web Development**: HTML, CSS, JavaScript, React
    - **Data Analysis**: Pandas, NumPy, Matplotlib
    - **Machine Learning**: Scikit-learn, TensorFlow, Keras
    - **Tools**: Git, Docker, Jenkins
    - **Others**: SQL, Oracle, Gradio
    """)

    # Achievements Section
    st.header(":blue[Achievements]")

    st.write("""
    - Got Elite in Data Analytics from NPTEL (2024)
    """)

    # Footer
    st.write("© 2024 Naveen S. All rights reserved.")

if __name__ == "__main__":
    main()
