import streamlit as st
import PyPDF2
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader


openai_api_key = st.secrets["OPENAI_API_KEY"]

def read_pdf(uploaded_files):
    text = ""

    for uploaded_file in uploaded_files:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(file):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.create_documents(file)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_documents(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to display the chat messages
def display_chat_message(message, is_user):
    if is_user:
        st.markdown("""
        <div class="user-bubble">
            <div>{}</div>
        </div>
        """.format(message), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="assistant-bubble">
            <div>{}</div>
        </div>
        """.format(message), unsafe_allow_html=True)


def main():
    # Page title
    st.set_page_config(page_title='Gforce Resume Assistant', layout='wide')
    st.title('Gforce Resume Assistant')

    # CSS Styles
    st.markdown("""
    <style>
        .chat-container {
            height: 400px;
            overflow-y: scroll;
            border: 1px solid #ddd;
            padding: 10px;
            margin-top: 10px;
        }
        .user-bubble {
            display: flex;
            justify-content: flex-start;
        }
        .user-bubble > div {
            padding: 15px;
            background-color: #e0e0e0;
            border-radius: 10px;
            width: 50%;
            margin-left: 50%;
        }
        .assistant-bubble {
            display: flex;
            justify-content: flex-end;
        }
        .assistant-bubble > div {
            padding: 15px;
            background-color: #0078d4;
            color: white;
            border-radius: 10px;
            width: 50%;
            margin-right: 50%;
        }
        .chat-input-prompt {
            position: sticky;
            bottom: 0;
            background-color: white;
            padding: 10px;
            width: 100%;
        }
        .chat-header {
            position: sticky;
            top: 0;
            background-color: #f2f2f2;
            padding: 10px;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

    if "conversation" not in st.session_state.keys():
        st.session_state.conversation = None
    if "chat_history" not in st.session_state.keys():
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
        display_chat_message(message.content, isinstance(message, HumanMessage))
        st.markdown('</div>', unsafe_allow_html=True)




    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = read_pdf(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
