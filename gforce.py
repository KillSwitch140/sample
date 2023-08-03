import streamlit as st
import os
import PyPDF2
import re
import spacy
import openai
from database import create_connection, create_resumes_table, insert_resume, get_all_resumes

# Set up your OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

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

# Ask the user for job details as soon as they upload resumes
job_title = st.sidebar.text_input("Enter the job title:")
qualifications = st.sidebar.text_area("Enter the qualifications for the job (separated by commas):")

# Display job details in the sidebar
st.sidebar.header('Job Details')
st.sidebar.write(f'Job Title: {job_title}')
st.sidebar.write(f'Qualifications: {qualifications}')

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

def generate_response(openai_api_key, query_text, candidates_info):
    # Load document if file is uploaded
    if len(candidates_info) > 0:
        # Prepare the conversation history with user query
        conversation_history = [{'role': 'user', 'content': query_text}]

        # Extract candidate names mentioned in the query
        mentioned_candidates = set()
        for candidate_info in candidates_info:
            candidate_name = candidate_info['name']
            if candidate_name and candidate_name.lower() in query_text.lower():
                mentioned_candidates.add(candidate_name)

        # Filter candidates based on the mentioned names
        filtered_candidates_info = [
            candidate_info
            for candidate_info in candidates_info
            if candidate_info['name'] in mentioned_candidates
        ]

        # Process each resume separately and store the summaries in filtered_candidates_info
        for idx, candidate_info in enumerate(filtered_candidates_info):
            resume_text = candidate_info["resume_text"]
            # Append the summarized resume text to the conversation history
            conversation_history.append({'role': 'system', 'content': f'Resume {idx + 1}: {resume_text}'})

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

def clear_conversation():
    # Clear the conversation history
    st.session_state.conversation_history = []

# Add a clear conversation button
if st.button('Clear Conversation'):
    clear_conversation()
