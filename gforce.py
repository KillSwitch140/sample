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

# Set up your OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

def read_pdf_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

# Custom CSS styling
st.markdown("""
<style>
/* Sticky top header */
.sticky-header {
    position: sticky;
    top: 0;
    background-color: #0078d4;
    color: white;
    padding: 8px;
    font-size: 18px;
    font-weight: bold;
}

/* Chat conversation area */
.chat-area {
    height: 400px;
    overflow-y: auto;
    padding: 8px;
    border-radius: 10px;
    background-color: #f2f2f2;
}

/* Chat bubbles */
.user-bubble {
    display: block;
    text-align: left;
    background-color: #e0e0e0;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    width: 70%;
}

.chatbot-bubble {
    display: block;
    text-align: right;
    background-color: #0078d4;
    color: white;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    width: 70%;
    margin-left: 30%;
}

/* Sticky bottom chat input prompt */
.sticky-input {
    position: sticky;
    bottom: 0;
    background-color: #f2f2f2;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)

# Page title and styling
st.set_page_config(page_title='GForce Resume Reader', layout='wide')
st.title('GForce Resume Reader')

# File upload
uploaded_file = st.file_uploader('Please upload your resume', type='pdf')

# Retrieve or initialize conversation history using SessionState
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Read the PDF content and set it as the initial context for the chatbot
if uploaded_file is not None:
    initial_context = read_pdf_text(uploaded_file)
    st.session_state.conversation_history = [{'role': 'system', 'content': initial_context}]

# User query
query_text = st.text_input('You (Type your message here):', value='', help='Ask away!', type='default')

# Form input and query
if st.button('Send', help='Click to submit the query'):
    if query_text.strip() != '':
        with st.spinner('Chatbot is typing...'):
            # Add the user query to the conversation history
            st.session_state.conversation_history.append({'role': 'user', 'content': query_text})
            # Get the updated conversation history
            conversation_history = st.session_state.conversation_history.copy()
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

# Display the entire conversation history in chat format
if st.session_state.conversation_history:
    st.header('Conversation History:')
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    for message in st.session_state.conversation_history:
        if message['role'] == 'user':
            st.markdown(f'<div class="user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        elif message['role'] == 'assistant':
            st.markdown(f'<div class="chatbot-bubble">{message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Add a clear conversation button
if st.button('Clear Conversation'):
    st.session_state.conversation_history.clear()

# Sticky input prompt
st.markdown('<div class="sticky-input">', unsafe_allow_html=True)
query_text = st.text_input('', value='', help='Type your message here...', type='default')
st.button('Send', help='Click to submit the query')
st.markdown('</div>', unsafe_allow_html=True)
