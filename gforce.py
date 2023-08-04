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
from qdrant_client import QdrantClient
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper


zapier_nla_api_key = st.secrets["ZAP_API_KEY"]
environ["ZAPIER_NLA_API_KEY"] = zapier_nla_api_key
openai_api_key = st.secrets["OPENAI_API_KEY"]

# client = QdrantClient(
#     url="https://fd3fb6ff-e014-4338-81ce-7d6e9db358b3.eu-central-1-0.aws.cloud.qdrant.io:6333", 
#     api_key=st.secrets["QDRANT_API_KEY"],
# )

# client.recreate_collection(
#     collection_name="resume_bot",
#     vectors_config=VectorParams(size=4, distance=Distance.DOT),
# )

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
    db = Chroma.from_documents(texts, embeddings)

    # Create retriever interface
    retriever = db.as_retriever(search_type="similarity")
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


llm = OpenAI(temperature=0)
zapier = ZapierNLAWrapper()
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
agent = initialize_agent(toolkit.get_tools(), llm, agent="zero-shot-react-description", verbose=True)


import streamlit as st

# Create a sidebar with text input boxes and a button
st.sidebar.header("Schedule Interview")
person_name = st.sidebar.text_input("Enter Person's Name", "")
person_email = st.sidebar.text_input("Enter Person's Email Address", "")
date = st.sidebar.date_input("Select Date for Interview")
time = st.sidebar.time_input("Select Time for Interview")
schedule_button = st.sidebar.button("Schedule Interview")

# Initialize a flag to check if the meeting has been successfully scheduled
meeting_scheduled = False

# Check if the button is clicked and the inputs are not empty
if schedule_button and person_name and person_email and date and time:
    # Create the combined string
    meeting_title = f"Hiring Plug Interview with {person_email}"
    date_time = f"{date} at {time}"
    schedule_meet = f"Schedule a virtual Google Meet titled {meeting_title} on {date_time}. Add this meeting as an event in my calendar"
    send_email = (
        f"Draft a well formatted, professional email to {person_email} notifying {person_name} that they have been selected\ "
        f"for an interview with Hiring Plug. Please search my calendar for Hiring Plug Interview with {person_email} and provide the respective meeting details, and ask if the "
        f"meeting timings are suitable for {person_name}."
        f"Dear [Candidate's Name],

Congratulations! You have been selected for an interview with Hiring Plug for the [Position Name] role.

Interview Details:
Date: [Date]
Time: [Time]
Location: Virtual (Google Meet)

We are excited to discuss your skills and qualifications further. Your interview details have been added to our calendar. If the provided timing is not suitable, please let us know, and we will try our best to accommodate.

We look forward to meeting you and learning more about your potential contributions to our team.

Best regards,
[Your Name]
Hiring Plug Team
"
    )

    # Execute the agent.run function for scheduling the meeting
    agent.run(schedule_meet)
    meeting_scheduled = True
    # Check if the meeting has been successfully scheduled
    time.sleep(5)
    agent.run(send_email)
    # Print or display the combined string
    st.sidebar.success("Interview Scheduled Successfully!")


    



