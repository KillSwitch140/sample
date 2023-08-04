import streamlit as st
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
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain import PromptTemplate
import pysqlite3
from langchain.chat_models import ChatOpenAI
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

openai_api_key = st.secrets["OPENAI_API_KEY"]

def read_pdf_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text
 

def generate_response(doc_texts, openai_api_key, query_text):
    input_dict = {
    "human_input": query_text, 
    "chat_history": "Previous conversation...",
    "context":  doc_texts
    }

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1,openai_api_key=openai_api_key)
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(doc_texts)
    
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)

    # Create retriever interface
    retriever = db.as_retriever()
    #Bot memory
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages="True",input_key=query_text)

    TEMPLATE = """You are a hiring manager's helpful assistant that reads multiple resumes of candidates and answers any questions related to the candidates,\
        You are chatbot that talks in a professional tone \
        Only answer the quesions truthfully and accurate do not provide further details.\
        If you don't know the answer, just say that you don't know, don't try to make up an answer.\
        If you are asked to summarize a candidate'sresume, summarize it in 5 sentences, 3 sentences for their experience and projects, 1 sentence for their education and 1 sentence for their skills\
        If you are asked to compare candidates just provide the summarization of their resumes
                        
        {context}

        The chat history so far: ```{chat_history}```

        The customer's latest message: ```{human_input}```
    """

    PROMPT_TEMPLATE = ChatPromptTemplate.from_template(TEMPLATE)
    prompt = PromptTemplate(
            input_variables=["chat_history", "human_input", "context"], template=TEMPLATE
            )
    # Create QA chain
    qa =RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        chain_type_kwargs={
            "prompt": prompt
        }
    )
    # Generate response
    response = qa.run(query_text)

    return response
    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi, I am your resume Q&A bot. How can I help you today?"}]

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

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
