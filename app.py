import streamlit as st
from streamlit_option_menu import option_menu
import firebase_admin
from firebase_admin import credentials, auth
from firebase_admin._auth_utils import UserNotFoundError
from streamlit_lottie import st_lottie
import importlib
import json
st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #2e86c1;
        font-family: 'Arial Black', Gadget, sans-serif;
    }
    .subtitle {
        font-size: 30px;
        color: #2874a6;
        font-family: 'Verdana', Geneva, sans-serif;
    }
    .author {
        font-size: 20px;
        color: #1b4f72;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    .upload-header {
        font-size: 25px;
        color: #1a5276;
        font-family: 'Georgia', serif;
    }
    .file-details {
        font-size: 20px;
        color: #154360;
        font-family: 'Courier New', Courier, monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate('streamlit-naveen-c3584545e869.json')
    firebase_admin.initialize_app(cred)

PAGES = {
    "Home": "home",
    "Data Collection":"Data",
    "File Preprocessing": "page1",
    "Analysis/Model Selection":"manual",
    "AutoML":"automl",
    "Predict":"predict",
    "About Me":"about"
    
}

def login(email, password):
    try:
        user = auth.get_user_by_email(email)
        st.success('Login Successful')
        st.session_state.username = user.display_name
        st.session_state.useremail = user.email
        st.session_state.signedout = True
        st.session_state.signout = True
        st.experimental_rerun()
    except UserNotFoundError:
        st.warning('Login Failed: User not found')
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')

def signup(email, password, username):
    try:
        user = auth.create_user(
            email=email,
            password=password,
            display_name=username
        )
        st.success('Account Created Successfully')
        st.markdown('Please login using your email and password')
        st.balloons()
    except Exception as e:
        st.warning(f'Sign Up Failed: {str(e)}')

def signout():
    st.session_state.signout = False
    st.session_state.signedout = False
    st.session_state.username = ''
    st.session_state.useremail = ''
    st.experimental_rerun()

def main():

    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'useremail' not in st.session_state:
        st.session_state.useremail = ''
    if 'signedout' not in st.session_state:
        st.session_state.signedout = False
    if 'signout' not in st.session_state:
        st.session_state.signout = False

    if not st.session_state.signedout:
        st.markdown("<div class='title'>MachinaMind: A sleek and powerful machine learning app.</div>", unsafe_allow_html=True)
        #st.markdown("<div class='subtitle'>Welcome to my page</div>", unsafe_allow_html=True)
        st.markdown("<div class='author'>by Naveen</div>", unsafe_allow_html=True)
        choice = st.selectbox('Login/Sign Up', ['Login', 'Sign Up'])

        if choice == 'Login':
            login_email = st.text_input('Email Address', key='login_email')
            login_password = st.text_input('Password', type='password', key='login_password')
            if st.button('Login'):
                login(login_email, login_password)
        else:
            signup_email = st.text_input('Email Address', key='signup_email')
            signup_password = st.text_input('Password', type='password', key='signup_password')
            signup_username = st.text_input('Enter your Unique Username', key='signup_username')
            if st.button('Create my account'):
                signup(signup_email, signup_password, signup_username)

    if st.session_state.signout:
        with st.sidebar:
            selected = option_menu(
                menu_title=f"Welcome, {st.session_state.username}",
                options=list(PAGES.keys()),
                icons=["house", "file-earmark","bi bi-database-fill-gear","bi bi-graph-up","bi bi-robot","bi bi-search","bi bi-person"],
                menu_icon="cast",
                default_index=0,
            )
        
        # Render the selected page content in the main section
        page = PAGES[selected]
        module = importlib.import_module(page)
        module.main()

        if st.sidebar.button('Sign Out'):
            signout()

if __name__ == '__main__':
    main()
