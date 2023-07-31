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

# Page title
st.set_page_config(page_title='GForce Resume Reader')
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
with st.form('myform', clear_on_submit=True):
    st.form_submit_button('Send', help='Click to submit the query')
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

# Display the entire conversation history with chat bubbles
if st.session_state.conversation_history:
    st.header('Conversation History:')
    chat_history_placeholder = st.empty()  # Placeholder to store chat history

    for message in st.session_state.conversation_history:
        if message['role'] == 'user':
            st.markdown(f'<div style="display: block; text-align: left; padding: 5px; background-color: #e0e0e0; border-radius: 10px; margin-bottom: 5px; width: 50%;">{message["content"]}</div>', unsafe_allow_html=True)
        elif message['role'] == 'assistant':
            st.markdown(f'<div style="display: block; text-align: right; padding: 5px; background-color: #0078d4; color: white; border-radius: 10px; margin-bottom: 5px; width: 50%;">{message["content"]}</div>', unsafe_allow_html=True)

# Add a clear conversation button
if st.button('Clear Conversation'):
    st.session_state.conversation_history.clear()
    
