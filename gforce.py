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


def read_pdf_text(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
    
def generate_response(uploaded_file, openai_api_key, conversation_history):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [read_pdf_text(uploaded_file)]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        
        # Handle multiple queries and store conversation history
        for user_query in conversation_history:
            response = qa.run(user_query)
            conversation_history.append((user_query, response))
        
        return conversation_history

# Page title
st.set_page_config(page_title='GForce Resume Reader')
st.title('GForce Resume Reader')

# File upload
uploaded_file = st.file_uploader('Please upload you resume', type='pdf')

# Retrieve or initialize conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# User query
query_text = st.text_input('How can I help?:', placeholder='Ask away!', disabled=not uploaded_file)

# Form input and query
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    submitted = st.form_submit_button('Submit', disabled=not (uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Loading response...'):
            # Add the user query to the conversation history
            st.session_state.conversation_history.append(query_text)
            # Get the updated conversation history
            conversation_history = st.session_state.conversation_history.copy()
            # Generate the response using the updated conversation history
            result = generate_response(uploaded_file, openai_api_key, conversation_history)
    else:
        result = []

# Display the conversation history
if result:
    st.header('Conversation History:')
    for user_query, response in result:
        st.subheader('User:')
        st.write(user_query)
        st.subheader('Chatbot:')
        st.write(response)
