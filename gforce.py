import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from htmlTemplates import css, bot_template, user_template

openai_api_key = st.secrets["OPENAI_API_KEY"]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents(documents)
    return texts

def get_vectorstore(texts):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    db = Chroma.from_documents(texts, embeddings)
    return db

def get_conversation_chain(db):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    retriever=db.as_retriever()
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        


from PyPDF2 import PdfReader
