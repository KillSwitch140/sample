import streamlit as st
import PyPDF2
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

api_key = st.secrets["OPENAI_API_KEY"]

def read_pdf_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def generate_response(doc_texts, openai_api_key, query_text):
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
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload
uploaded_files = st.file_uploader('Upload one or more PDF documents', type='pdf', accept_multiple_files=True)

# Query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', key="query_text")

# Convert uploaded PDFs to text
doc_texts = []
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        text = read_pdf_text(uploaded_file)
        doc_texts.append(text)

# Chat history display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Form input and query
if doc_texts and query_text:
    result = []
    with st.form('myform', clear_on_submit=True):
        submitted = st.form_submit_button('Submit')
        if submitted:
            with st.spinner('Calculating...'):
                response = generate_response(doc_texts, openai_api_key, query_text)
                result.append(response)
                st.session_state.messages.append({"role": "user", "content": query_text})
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.warning("Please upload one or more PDF documents and enter a question to start the conversation.")

# Clear chat history button
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
