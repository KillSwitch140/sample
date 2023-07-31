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
st.title('💬 GForce Resume Reader')


# File upload
uploaded_file = st.file_uploader('Please upload your resume', type='pdf')

# Retrieve or initialize conversation history using SessionState
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

# Read the PDF content and set it as the initial context for the chatbot
if uploaded_file is not None:
    initial_context = read_pdf_text(uploaded_file)
    st.session_state.messages[0]["content"] = initial_context

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.text_area("You:", value=msg["content"], key="user_input", height=50)
    else:
        st.text_area("Assistant:", value=msg["content"], key="assistant_output", height=50)

# User query
user_input = st.text_input('Type your message here:', value='')

# Form input and query
if user_input.strip() != '':
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages,
        api_key=openai_api_key
    )
    msg = response.choices[0].message
    st.session_state.messages[-1]["content"] = user_input
    st.session_state.messages.append(msg)
    st.session_state.messages = st.session_state.messages[-5:]  # Limiting chat history to last 5 messages

# Add a clear conversation button
if st.button('Clear Conversation'):
    st.session_state.messages.clear()
