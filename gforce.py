import streamlit as st
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

# Function to read PDF text
def read_pdf_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

# Function to extract candidate name using spaCy NER
def extract_candidate_name(resume_text):
    # Assume the candidate name is in the first line of the resume text
    first_line = resume_text.strip().split('\n')[0]
    
    # Initialize spaCy NER model
    nlp = spacy.load("en_core_web_sm")
    
    # Process the first line with spaCy NER
    doc = nlp(first_line)
    candidate_name = None
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            candidate_name = ent.text
            break

    # If spaCy NER did not find a PERSON entity in the first line, use the entire first line as the candidate name
    if not candidate_name:
        candidate_name = first_line.strip()
        
    return candidate_name

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [
        {'role': 'system', 'content': 'Hello! I am your recruiter assistant. My role is to go through resumes and help recruiters make informed decisions.'}
    ]

# Page title and styling
st.set_page_config(page_title='GForce Resume Reader', layout='wide')
st.title('GForce Resume Reader')

# List to store uploaded resume contents and extracted information
uploaded_resumes = []
candidates_info = []

# File upload
uploaded_files = st.file_uploader('Please upload your resume', type='pdf', accept_multiple_files=True)

# Ask the user for job details as soon as they upload resumes
job_title = st.sidebar.text_input("Enter the job title:")
qualifications = st.sidebar.text_area("Enter the qualifications for the job (separated by commas):")

# Display job details in the sidebar
st.sidebar.header('Job Details')
st.sidebar.write(f'Job Title: {job_title}')
st.sidebar.write(f'Qualifications: {qualifications}')

# Process uploaded resumes and store in the database
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            resume_text = read_pdf_text(uploaded_file)
            uploaded_resumes.append(resume_text)
            # Extract candidate name using spaCy NER
            candidate_name = extract_candidate_name(resume_text)
            # Store the information for each candidate
            candidate_info = {
                'name': candidate_name,
                'resume_text': resume_text
            }
            candidates_info.append(candidate_info)
            # Store the resume and information in the database
            insert_resume(connection, candidate_info)

# Function to prompt GPT-3.5-turbo with job details and user query
def generate_response(openai_api_key, job_title, qualifications, user_query, candidates_info, connection):
    if "gpa" in user_query.lower():
        candidate_name = extract_candidate_name(user_query)
        if candidate_name:
            # Query the database to get the candidate's GPA
            query = f"SELECT gpa FROM resumes WHERE name = '{candidate_name}'"
            cursor = connection.cursor()
            cursor.execute(query)
            gpa_result = cursor.fetchone()
            cursor.close()

            if gpa_result:
                # The gpa_result is a tuple with a single element (the GPA value)
                gpa = gpa_result[0]
                response = f"The GPA for {candidate_name} is {gpa}."
            else:
                response = f"Sorry, the GPA for {candidate_name} is not available."

        else:
            response = "Sorry, I couldn't find the candidate's name to fetch the GPA."

    elif "email" in user_query.lower():
        candidate_name = extract_candidate_name(user_query)
        if candidate_name:
            # Query the database to get the candidate's email
            query = f"SELECT email FROM resumes WHERE name = '{candidate_name}'"
            cursor = connection.cursor()
            cursor.execute(query)
            email_result = cursor.fetchone()
            cursor.close()

            if email_result:
                # The email_result is a tuple with a single element (the email value)
                email = email_result[0]
                response = f"The email for {candidate_name} is {email}."
            else:
                response = f"Sorry, the email for {candidate_name} is not available."

        else:
            response = "Sorry, I couldn't find the candidate's name to fetch the email."

    else:
        # Rest of the code to handle other queries (without GPA or email extraction) remains the same...
        # Use GPT-3.5-turbo for recruiter assistant tasks based on prompts
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            api_key=openai_api_key
        )

        response = response['choices'][0]['message']['content']

    return response

# User query
user_query = st.text_area('You (Type your message here):', value='', help='Ask away!', height=100, key="user_input")


# Form input and query
send_user_query = st.button('Send', help='Click to submit the query', key="send_user_query")
if send_user_query:
    if user_query.strip() != '':
        with st.spinner('Chatbot is typing...'):
            # Generate the response using the job details and user query
            response = generate_response(openai_api_key, job_title, qualifications, user_query, candidates_info, connection)
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

