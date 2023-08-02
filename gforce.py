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
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
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

def extract_experience_dates(resume_text):
    # Use regular expressions to extract experience dates from the resume text
    date_pattern = r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"  # Assumes dates in the format DD/MM/YYYY
    experience_dates = re.findall(date_pattern, resume_text)

    # Convert the extracted dates to datetime objects
    experience_dates = [datetime(int(year), int(month), int(day)) for day, month, year in experience_dates]

    return experience_dates



# Function to summarize the text using Cohere API
def summarize_text(text):
    # Use a text summarization model to summarize the text within the specified token limit.
    co = cohere.Client(cohere_api_key)
    summarized_text = co.summarize(
        model='summarize-medium', 
        length='long',
        extractiveness='high',
        format='paragraph',
        temperature= 0.2,
        additional_command='Generate a summary for this resume',
        text=text
    )
    return summarized_text

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
            # Calculate years of experience
            experience_dates = extract_experience_dates(resume_text)
            # Check if there are any experience dates before calculating the oldest and latest
            if experience_dates:
                oldest_experience_date = min(experience_dates)
                latest_experience_date = max(experience_dates)
                years_of_experience = (latest_experience_date - oldest_experience_date).days / 365
            else:
                # If no experience dates are found, set years_of_experience to None
                years_of_experience = None
            # Summarize the resume text
            summarized_resume_text = summarize_text(resume_text)
            # Store the information for each candidate
            candidate_info = {
                'name': candidate_name,
                'gpa': gpa,
                'email': email,
                'resume_text': resume_text,
                'summarized_resume_text': summarized_resume_text,
                'years_of_experience': years_of_experience
            }
            candidates_info.append(candidate_info)
            # Store the resume and information in the database
            insert_resume(connection, candidate_info)

# Function to get vector embeddings using Cohere API
def get_vector_embedding(text):
    co = cohere.Client(cohere_api_key)
    response = co.embed(
        texts=[text],
        model='embed-english-v2.0',
    )
    embedding = response['embeddings'][0]
    return embedding

# Calculate and store vector embeddings for each candidate
for candidate_info in candidates_info:
    text = candidate_info["summarized_resume_text"]
    embedding = get_vector_embedding(text)
    # Store the embedding in the database
    update_embeddings(connection, candidate_info["name"], embedding)

# Function to retrieve vector embeddings from the database
def get_candidate_embedding(candidate_name):
    cursor = connection.cursor()
    cursor.execute("SELECT embedding FROM resumes WHERE name=?", (candidate_name,))
    result = cursor.fetchone()
    cursor.close()
    if result:
        return np.frombuffer(result[0])
    return None

# Update the generate_response function to use vector embeddings from the database
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

        # Use vector embeddings to represent desired qualifications and experience levels
        desired_qualifications = "Linux, React, MVP"  # Replace this with the desired qualifications
        desired_experience = 3  # Replace this with the desired minimum years of experience

        # Convert the desired qualifications and experience levels to vector representations
        desired_qualifications_vector = get_vector_embedding(desired_qualifications)
        desired_experience_vector = np.array([desired_experience])

        # Calculate the similarity score for each candidate based on qualifications and experience
        similarity_scores = []
        for candidate_info in candidates_info:
            candidate_embedding = get_candidate_embedding(candidate_info["name"])
            if candidate_embedding is None:
                continue

            # Convert candidate's qualifications and experience to vector representations
            candidate_qualifications_vector = candidate_embedding
            candidate_experience_vector = np.array([candidate_info["years_of_experience"]])

            # Calculate the cosine similarity between the candidate and desired vectors
            qualification_similarity = 1 - cosine(desired_qualifications_vector, candidate_qualifications_vector)
            experience_similarity = 1 - cosine(desired_experience_vector, candidate_experience_vector)

            # Combine qualification and experience similarity scores
            overall_similarity = (qualification_similarity + experience_similarity) / 2
            similarity_scores.append(overall_similarity)

        # Find the top candidates based on similarity scores
        num_top_candidates = 3  # You can choose the number of top candidates to display
        top_candidate_indices = np.argsort(similarity_scores)[-num_top_candidates:]
        top_candidates = [candidates_info[idx] for idx in top_candidate_indices][::-1]

        # Generate the response with the top candidates
        response = f"Top {num_top_candidates} candidates based on qualifications and experience:\n"
        for rank, candidate in enumerate(top_candidates, 1):
            response += f"{rank}. {candidate['name']} - Qualification Score: {similarity_scores[top_candidate_indices[rank - 1]]:.2f}\n"

        return response

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
            response = generate_response(openai_api_key, user_query, conversation_history)
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
