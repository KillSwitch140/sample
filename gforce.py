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
import re
import spacy
import cohere
import sqlite3
from database import create_connection, create_resumes_table, insert_resume, get_all_resumes


# Set up your OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
cohere_api_key = st.secrets["COHERE_API_KEY"]
# Connect to the database and create the table
database_name = "resumes.db"
connection = create_connection(database_name)
create_resumes_table(connection)


def read_pdf_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

# Function to extract GPA using regular expression
def extract_gpa(text):
    gpa_pattern = r"\bGPA\b\s*:\s*([\d.]+)"
    gpa_match = re.search(gpa_pattern, text, re.IGNORECASE)
    return gpa_match.group(1) if gpa_match else None

# Function to extract email using regular expression
def extract_email(text):
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    email_match = re.search(email_pattern, text)
    return email_match.group() if email_match else None

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Function to extract candidate name using spaCy NER
def extract_candidate_name(resume_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(resume_text)
    candidate_name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            candidate_name = ent.text
            break
    return candidate_name



# Page title and styling
st.set_page_config(page_title='GForce Resume Reader', layout='wide')
st.title('GForce Resume Reader')

# List to store uploaded resume contents and extracted information
uploaded_resumes = []
candidates_info = []

# File upload
uploaded_files = st.file_uploader('Please upload your resume', type='pdf', accept_multiple_files=True)

# Process uploaded resumes and store in the database
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            resume_text = read_pdf_text(uploaded_file)
            uploaded_resumes.append(resume_text)
            # Extract GPA, email, and past
            gpa = extract_gpa(resume_text)
            email = extract_email(resume_text)
            # Extract candidate name using spaCy NER
            candidate_name = extract_candidate_name(resume_text)
            # Store the information for each candidate
            candidate_info = {
                'name': candidate_name,
                'gpa': gpa,
                'email': email,
                'resume_text': resume_text
            }
            candidates_info.append(candidate_info)
            # Store the resume and information in the database
            insert_resume(connection, candidate_info)


def summarize_text(text):
    # Use a text summarization model to summarize the text within the specified token limit.
    co = cohere.Client(cohere_api_key)
    summarized_text = co.summarize(
        model='summarize-xlarge', 
        length='long',
        extractiveness='high',
        format='paragraph',
        temperature= 0.2,
        text=text,
        additional_command = 'Generate a summary for this resume'
    )
    return summarized_text

def generate_response(openai_api_key, query_text, candidates_info):
    # Load document if file is uploaded
    if len(candidates_info) > 0:
        # Prepare the conversation history with user query
        conversation_history = [{'role': 'user', 'content': query_text}]

        # Process each resume separately and store the summaries in candidates_info
        for idx, candidate_info in enumerate(candidates_info):
            resume_text = candidate_info["resume_text"]
            # Summarize each resume text to fit within the token limit
            max_tokens = 4096  # Adjust this token limit as needed
            summarized_resume_text = summarize_text(resume_text)
            candidates_info[idx]["summarized_resume_text"] = summarized_resume_text

            # Append the summarized resume text to the conversation history
            conversation_history.append({'role': 'system', 'content': f'Resume {idx + 1}: {summarized_resume_text}'})

       # Use GPT-3.5-turbo for recruiter assistant tasks based on prompts
        recruiter_prompts = {
            "compare_candidates": "Please compare the candidates based on their qualifications and experience.",
            "top_candidates": "Can you suggest the top candidates for the position based on if they possess a minimum of 3 years of experience in Linux, React, MVP, etc. Or similar qualifications",
            "identify_strengths": "Identify the strengths of each candidate and their suitability for the role.",
            "evaluate_experience": "Evaluate the past experiences of the candidates and their relevance to the job.",
        }

        # Add a prompt for the specific query provided by the user
        if query_text in recruiter_prompts:
            conversation_history.append({'role': 'user', 'content': recruiter_prompts[query_text]})

        # Generate the response using the updated conversation history
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            api_key=openai_api_key
        )
        # Get the assistant's response
        assistant_response = response['choices'][0]['message']['content']
        return assistant_response

    else:
        return "Sorry, no resumes found in the database. Please upload resumes first."


# User query
user_query = st.text_area('You (Type your message here):', value='', help='Ask away!', height=100, key="user_input")

# Form input and query
send_user_query = st.button('Send', help='Click to submit the query', key="send_user_query")
if send_user_query:
    if user_query.strip() != '':
        with st.spinner('Chatbot is typing...'):
            # Add the user query to the conversation history
            st.session_state.conversation_history.append({'role': 'user', 'content': user_query})
            # Get the updated conversation history
            conversation_history = st.session_state.conversation_history.copy()
            # Append the uploaded resumes' content to the conversation history
            conversation_history.extend([{'role': 'system', 'content': resume_text} for resume_text in uploaded_resumes])
            # Generate the response using the updated conversation history
            response = generate_response(openai_api_key, user_query, candidates_info)
            # Append the assistant's response to the conversation history
            st.session_state.conversation_history.append({'role': 'assistant', 'content': response})


# Chat UI with sticky headers and input prompt
st.markdown("""
<style>
    .chat-container {
        height: 25px;
        overflow-y: scroll;
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

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display the entire conversation history in chat format
if st.session_state.conversation_history:
    for i, message in enumerate(st.session_state.conversation_history):
        if message['role'] == 'user':
            st.markdown(f'<div class="user-bubble"><div>{message["content"]}</div></div>', unsafe_allow_html=True)
        elif message['role'] == 'assistant':
            st.markdown(f'<div class="assistant-bubble"><div>{message["content"]}</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Add a clear conversation button
clear_conversation = st.button('Clear Conversation', key="clear_conversation")
if clear_conversation:
    st.session_state.conversation_history.clear()
