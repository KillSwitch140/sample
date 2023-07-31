import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import PyPDF2
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import openai
import spacy
import subprocess

# Set up your OpenAI API key
openai_api_key = "YOUR_OPENAI_API_KEY"

def download_spacy_model():
    try:
        subprocess.check_output(["python", "-m", "spacy", "download", "en_core_web_sm"])
    except subprocess.CalledProcessError as e:
        st.error("Failed to download the spaCy model.")
        st.stop()

# Check if the spaCy model is installed, if not, download it
if "en_core_web_sm" not in spacy.util.get_installed_models():
    st.info("Downloading spaCy model...")
    download_spacy_model()
    st.info("Download complete. You can now proceed.")
else:
    nlp = spacy.load("en_core_web_sm"))

def read_pdf_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def extract_information(text):
    doc = nlp(text)
    email_addresses = []
    gpa = None
    schools = []
    previous_companies = []

    for ent in doc.ents:
        if ent.label_ == "EMAIL":
            email_addresses.append(ent.text)
        elif ent.label_ == "GPA":
            gpa = ent.text
        elif ent.label_ == "ORG" and "school" in ent.text.lower():
            schools.append(ent.text)
        elif ent.label_ == "ORG" and "company" in ent.text.lower():
            previous_companies.append(ent.text)

    return email_addresses, gpa, schools, previous_companies

# Page title and styling
st.set_page_config(page_title='GForce Resume Reader', layout='wide')
st.title('GForce Resume Reader')

# List to store uploaded resume contents
uploaded_resumes = []

# File upload
uploaded_files = st.file_uploader('Please upload your resume', type='pdf', accept_multiple_files=True)

# Process uploaded resumes
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            uploaded_resumes.append(read_pdf_text(uploaded_file))

# Retrieve or initialize conversation history using SessionState
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# User query
user_query = st.text_area('You (Type your message here):', value='', help='Ask away!', height=100, key="user_input")

# Form input and query
send_user_query = st.button('Send', help='Click to submit the query', key="send_user_query")
if send_user_query:
    if user_query.strip() != '':
        with st.spinner('Chatbot is typing...'):
            # Add the user query to the conversation history
            st.session_state.conversation_history.append({'role': 'user', 'content': user_query})
            # Get the updated conversation history
            conversation_history = st.session_state.conversation_history.copy()
            # Append the uploaded resumes' content to the conversation history
            conversation_history.extend([{'role': 'system', 'content': resume_text} for resume_text in uploaded_resumes])
            # Generate the response using the updated conversation history
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation_history,
                api_key=openai_api_key
            )
            # Get the assistant's response
            assistant_response = response['choices'][0]['message']['content']
            # Append the assistant's response to the conversation history
            st.session_state.conversation_history.append({'role': 'assistant', 'content': assistant_response})

# Chat UI with sticky headers and input prompt
st.markdown("""
<style>
    .chat-container {
        height: 35px;
        overflow-y: scroll;
    }
    .user-bubble {
        display: flex;
        justify-content: flex-start;
    }
    .user-bubble > div {
        padding: 15px;
        background-color: #e0e0e0;
        border-radius: 10px;
        width: 50%;
        margin-left: 50%;
    }
    .assistant-bubble {
        display: flex;
        justify-content: flex-end;
    }
    .assistant-bubble > div {
        padding: 15px;
        background-color: #0078d4;
        color: white;
        border-radius: 10px;
        width: 50%;
        margin-right: 50%;
    }
    .chat-input-prompt {
        position: sticky;
        bottom: 0;
        background-color: white;
        padding: 10px;
        width: 100%;
    }
    .chat-header {
        position: sticky;
        top: 0;
        background-color: #f2f2f2;
        padding: 10px;
        width: 100%;
    }s
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display the entire conversation history in chat format
if st.session_state.conversation_history:
    for i, message in enumerate(st.session_state.conversation_history):
        if message['role'] == 'user':
            st.markdown(f'<div class="user-bubble"><div>{message["content"]}</div></div>', unsafe_allow_html=True)
        elif message['role'] == 'assistant':
            st.markdown(f'<div class="assistant-bubble"><div>{message["content"]}</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Add a clear conversation button
clear_conversation = st.button('Clear Conversation', key="clear_conversation")
if clear_conversation:
    st.session_state.conversation_history.clear()

# Extract and display information from uploaded resumes
if uploaded_resumes:
    st.header('Extracted Information from Resumes')
    for i, resume_text in enumerate(uploaded_resumes):
        st.subheader(f'Resume {i + 1}')
        st.write(resume_text)

        email_addresses, gpa, schools, previous_companies = extract_information(resume_text)

        if email_addresses:
            st.write(f'Email Addresses: {", ".join(email_addresses)}')
        if gpa:
            st.write(f'GPA: {gpa}')
        if schools:
            st.write(f'Schools: {", ".join(schools)}')
        if previous_companies:
            st.write(f'Previous Companies: {", ".join(previous_companies)}')
