import streamlit as st
import openai
import PyPDF2

# Set up your OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Function to read PDF text
def read_pdf_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [{'role': 'system', 'content': 'Hello! I am your recruiter assistant. My role is to go through resumes and help recruiters make informed decisions.'}]


# Function to prompt GPT-3.5-turbo with user query
def generate_response(openai_api_key, user_query, conversation_history):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history + [{'role': 'user', 'content': user_query}],
        api_key=openai_api_key
    )

    return response['choices'][0]['message']['content']

# Page title and styling
st.set_page_config(page_title='GForce Resume Reader', layout='wide')
st.title('GForce Resume Reader')

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [{'role': 'system', 'content': 'Hello! I am your recruiter assistant. My role is to go through resumes and help recruiters make informed decisions.'}]

# Sidebar for job details
st.sidebar.title('Job Details')
job_title = st.sidebar.text_input("Enter the job title:")
qualifications = st.sidebar.text_area("Enter the qualifications for the job (separated by commas):")

# File upload
uploaded_files = st.file_uploader('Please upload your resume', type='pdf', accept_multiple_files=True)

# User query
user_query = st.text_area('You (Type your message here):', value='', help='Ask away!', height=100, key="user_input")

# Form input and query
send_user_query = st.button('Send', help='Click to submit the query', key="send_user_query")
if send_user_query:
    if user_query.strip() != '':
        with st.spinner('Chatbot is typing...'):
            # Generate the response using the user query
            response = generate_response(openai_api_key, user_query, st.session_state.conversation_history)
            # Append the user's and assistant's messages to the conversation history
            st.session_state.conversation_history.append({'role': 'user', 'content': user_query})
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
