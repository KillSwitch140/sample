import streamlit as st
import PyPDF2
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pysqlite3
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

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(doc_texts)

    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)

    # Create retriever interface
    retriever = db.as_retriever()

    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)

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

# Clear chat history button
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
