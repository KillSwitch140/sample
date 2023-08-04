import streamlit as st
import datetime
import os
from os import environ
import PyPDF2
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pysqlite3
from langchain.chat_models import ChatOpenAI
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import qdrant_client
from qdrant_client import QdrantClient,models
from qdrant_client.http.models import PointStruct
from langchain.agents import initialize_agent
from langchain.vectorstores import Qdrant

openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ['QDRANT_COLLECTION'] ="resume"


client = QdrantClient(
    url="https://fd3fb6ff-e014-4338-81ce-7d6e9db358b3.eu-central-1-0.aws.cloud.qdrant.io:6333", 
    api_key=st.secrets["QDRANT_API_KEY"],
)
collection_config = qdrant_client.http.models.VectorParams(
        size=1536,
        distance=qdrant_client.http.models.Distance.COSINE
    )
client.recreate_collection(
   collection_name=os.getenv("QDRANT_COLLECTION"),
    vectors_config=collection_config)



def read_pdf_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def generate_response(doc_texts, openai_api_key, query_text):

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1,openai_api_key=openai_api_key)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(doc_texts)
    
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Create a vectorstore from documents
    qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url="https://fd3fb6ff-e014-4338-81ce-7d6e9db358b3.eu-central-1-0.aws.cloud.qdrant.io:6333",
    prefer_grpc=True,
    api_key=st.secrets["QDRANT_API_KEY"],
    collection_name="resume",
)
    # Create retriever interface
    retriever = qdrant.as_retriever(search_type="similarity")
    #Bot memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # template  = """
    #         You are an AI assistant created to help hiring managers review resumes and shortlist candidates. You have been provided with resumes and job descriptions to review. When asked questions, use the provided documents to provide helpful and relevant information to assist the hiring manager. Be concise, polite and professional. Do not provide any additional commentary or opinions beyond answering the questions directly based on the provided documents.
    #         Question:{query}
    # """
    # QA_CHAIN_PROMPT = PromptTemplate.from_template(template,input_variables=['query'])
    # QA_CHAIN_PROMPT.format(query= query_text)
    #Create QA chain 
    qa = ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever,memory=memory)
    response = qa.run(query_text)
    
    return response
    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "You are a AI assistant created to help hiring managers review resumes and shortlist candidates. You have been provided with resumes and job descriptions to review. When asked questions, use the provided documents to provide helpful and relevant information to assist the hiring manager. Be concise, polite and professional. Do not provide any additional commentary or opinions beyond answering the questions directly based on the provided documents."}]

# Page title
st.set_page_config(page_title='Gforce Resume Assistant', layout='wide')
st.title('Gforce Resume Assistant')

# File upload
uploaded_files = st.file_uploader('Upload PDF(s)', type=['pdf'], accept_multiple_files=True)

# Query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.')

# Initialize chat placeholder as an empty list
if "chat_placeholder" not in st.session_state.keys():
    st.session_state.chat_placeholder = []

# Form input and query
if st.button('Submit', key='submit_button'):
    if openai_api_key.startswith('sk-'):
        if uploaded_files and query_text:
            documents = [read_pdf_text(file) for file in uploaded_files]
            with st.spinner('Chatbot is typing...'):
                response = generate_response(documents, openai_api_key, query_text)
                st.session_state.chat_placeholder.append({"role": "user", "content": query_text})
                st.session_state.chat_placeholder.append({"role": "assistant", "content": response})

            # Update chat display
            for message in st.session_state.chat_placeholder:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        else:
            st.warning("Please upload one or more PDF files and enter a question to start the conversation.")

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.chat_placeholder = []
    uploaded_files.clear()
    query_text = ""
    st.empty()  # Clear the chat display

st.button('Clear Chat History', on_click=clear_chat_history)

# Create a sidebar with text input boxes and a button
st.sidebar.header("Schedule Interview")
person_name = st.sidebar.text_input("Enter Person's Name", "")
person_email = st.sidebar.text_input("Enter Person's Email Address", "")
date = st.sidebar.date_input("Select Date for Interview")
time = st.sidebar.time_input("Select Time for Interview")
schedule_button = st.sidebar.button("Schedule Interview")

if schedule_button:
    if not person_name:
        st.sidebar.error("Please enter the person's name.")
    elif not person_email:
        st.sidebar.error("Please enter the person's email address.")
    elif not date:
        st.sidebar.error("Please select the date for the interview.")
    elif not time:
        st.sidebar.error("Please select the time for the interview.")
    else:
        # Call the schedule_interview function from the zap.py file
        success = schedule_interview(person_name, person_email, date, time)

        if success:
            st.sidebar.success("Interview Scheduled Successfully!")
        else:
            st.sidebar.error("Failed to Schedule Interview")
