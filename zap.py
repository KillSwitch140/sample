from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
import os
import time


zapier_nla_api_key = st.secrets["ZAP_API_KEY"]
environ["ZAPIER_NLA_API_KEY"] = zapier_nla_api_key
openai_api_key = st.secrets["OPENAI_API_KEY"]

llm = OpenAI(temperature=0)
zapier = ZapierNLAWrapper()
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
agent = initialize_agent(toolkit.get_tools(), llm, agent="zero-shot-react-description", verbose=True)

def schedule_interview(person_name, person_email, date, time):
    # Create the combined string
    meeting_title = f"Hiring Plug Interview with {person_email}"
    date_time = f"{date} at {time}"
    schedule_meet = f"Schedule a 30 min virtual Google Meet titled {meeting_title} on {date_time}. Add the created meeting's details as a new event in my calendar"
    send_email = (
        f"Draft a well formatted, professional email to {person_email} notifying {person_name} that they have been selected "
        f"for an interview with Hiring Plug. Please search my calendar for 'Hiring Plug Interview with {person_name}' and provide the respective meeting details."
    )

    # Execute the agent.run function for scheduling the meeting
    agent.run(schedule_meet)
    time.sleep(5)  # Add a 5-second delay
    # Execute the agent.run function for sending the email
    agent.run(send_email)

    return True  # Return True if the interview is scheduled and the email is sent successfully
