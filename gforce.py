import streamlit as st
import os
import PyPDF2
import re
import spacy
import openai
from transformers import pipeline
from database import create_connection, create_resumes_table, insert_resume, get_all_resumes, get_candidate_info_from_database

# Set up your OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Connect to the database and create the table
database_name = "resumes.db"
connection = create_connection(database_name)
create_resumes_table(connection)

# Load NER and NEL models
nlp_ner = spacy.load("en_core_web_sm")

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


def read_pdf_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def extract_gpa(text):
    gpa_pattern = r"\bGPA\b\s*:\s*([\d.]+)"
    gpa_match = re.search(gpa_pattern, text, re.IGNORECASE)
    return gpa_match.group(1) if gpa_match else None

def extract_email(text):
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    email_match = re.search(email_pattern, text)
    return email_match.group() if email_match else None

def extract_candidate_name(resume_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(resume_text)
    candidate_name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            candidate_name = ent.text
            break
    return candidate_name

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

def extract_named_entities(text):
    doc = nlp_ner(text)
    named_entities = []
    for ent in doc.ents:
        named_entities.append({
            "text": ent.text,
            "label": ent.label_
        })
    return named_entities

def extract_named_entity_links(text):
    # Simple keyword-based approach for Named Entity Linking (NEL)
    entities = [
        {"word": "John Doe", "entity": "PERSON", "uri": "https://example.com/john-doe"},
        {"word": "Google", "entity": "ORG", "uri": "https://example.com/google"},
        # Add more entities and URLs as needed
    ]
    named_entity_links = []
    for ent in entities:
        if ent['word'].lower() in text.lower():
            named_entity_links.append(ent)
    return named_entity_links

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
    if len(candidates_info) > 0:
        conversation_history = [{'role': 'user', 'content': query_text}]
        candidate_name = extract_candidate_name(query_text)

        if 'compare' in query_text.lower():
            conversation_history.append({'role': 'system', 'content': f'Candidate 1: {candidates_info[0]["name"]}'})
            conversation_history.append({'role': 'system', 'content': f'Candidate 2: {candidates_info[1]["name"]}'})
        
        elif 'email' in query_text.lower() or 'gpa' in query_text.lower() or 'past experience' in query_text.lower():
            conversation_history.append({'role': 'system', 'content': f'Candidate: {candidate_name}'})

            candidate_info = get_candidate_info_from_database(candidate_name)

            conversation_history.append({'role': 'system', 'content': f'Candidate Info: {candidate_info}'})

            named_entities = extract_named_entities(candidate_info['resume_text'])
            named_entity_links = extract_named_entity_links(candidate_info['resume_text'])
            conversation_history.append({'role': 'system', 'content': f'Named Entities: {named_entities}'})
            conversation_history.append({'role': 'system', 'content': f'Named Entity Links: {named_entity_links}'})
            
            if 'experience' in query_text.lower():
                conversation_history.append({'role': 'system', 'content': f'Candidate: {candidate_name}, tell me about your past experience.'})
        
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation_history,
                api_key=openai_api_key
            )
            assistant_response = response['choices'][0]['message']['content']
            return assistant_response
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            api_key=openai_api_key
        )
        assistant_response = response['choices'][0]['message']['content']

        if candidate_name and candidate_name not in assistant_response:
            corrected_response = f"I apologize, I provided information about the wrong candidate. Let me clarify. {candidate_name}'s " + assistant_response
            return corrected_response

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
