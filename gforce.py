import streamlit as st
import PyPDF2
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

def read_pdf(uploaded_files):
    text = ""

    for uploaded_file in uploaded_files:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            text += page_text
            # Debug output to check page text
            st.write(f"Page Text for {uploaded_file.name}: {page_text}")

    return text

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents(documents)
    return texts

def get_vectorstore(texts):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl)
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
        


def main():
    # Page title
    st.set_page_config(page_title='Gforce Resume Assistant', layout='wide')
    st.title('Gforce Resume Assistant')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader('Upload PDF(s)', type=['pdf'], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Debug output to check uploaded PDFs
                if pdf_docs:
                    st.write(f"Number of uploaded PDFs: {len(pdf_docs)}")
                    for pdf_doc in pdf_docs:
                        st.write(f"Uploaded PDF: {pdf_doc.name}")
                else:
                    st.warning("No PDFs uploaded.")
                    return

                # get pdf text
                raw_text = read_pdf(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)



if __name__ == '__main__':
    main()
