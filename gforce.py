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

# Initialize or retrieve conversation history using Streamlit session_state
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

# Read the PDF content and set it as context for the chatbot
if uploaded_file is not None:
    resume_text = read_pdf_text(uploaded_file)
    if resume_text.strip() != "":
        st.session_state.messages[0]["content"] = f"Your resume:\n{resume_text}"

# Display chat history
st.title("💬 Chatbot")
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input and chatbot response
if prompt := st.text_input("You (Type your message here):"):
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages, api_key=openai_api_key)
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("Assistant").write(msg.content)
