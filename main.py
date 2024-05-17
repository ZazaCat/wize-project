import streamlit as st
import requests
import json
import os

class Chatbot:
    def __init__(self):
        self.api_url = "https://omniplex.ai/api/chat"  # Update the API URL to the correct endpoint if necessary
        self.model = "gpt-4o"
        self.temperature = 1
        self.max_tokens = 4096
        self.top_p = 1
        self.system_prompt = "You are a helpful AI assistant."

    def send_message(self, chat_history, message):
        payload = {
            "messages": [{"role": "system", "content": self.system_prompt}] + chat_history,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": True
        }
        with st.spinner("Thinking..."):
            response = requests.post(self.api_url, headers={'Content-Type': 'application/json'}, json=payload, stream=True)
        if response.status_code == 200:
            response_text = ""
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    chunk_str = chunk.decode('utf-8')
                    response_text += chunk_str
                    yield chunk_str
            chat_history.append({"role": "assistant", "content": response_text})
        else:
            yield f"Error: {response.status_code}"

# File path for storing the conversation data
DATA_FILE = "/sdcard/Download/conversations.json"

def save_data():
    """ Save the conversation data to a JSON file. """
    try:
        with open(DATA_FILE, "w") as file:
            json.dump({"users": st.session_state.users, "conversations": st.session_state.all_conversations}, file)
    except IOError as e:
        st.error(f"Failed to save data: {e}")

def load_data():
    """ Load the conversation data from a JSON file. """
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as file:
                data = json.load(file)
                if isinstance(data, dict):
                    st.session_state.users = data.get("users", {})
                    st.session_state.all_conversations = data.get("conversations", {})
                else:
                    raise json.JSONDecodeError("Not a dictionary", "", 0)
        except (json.JSONDecodeError, IOError) as e:
            st.session_state.users = {}
            st.session_state.all_conversations = {}
            st.error(f"Error loading conversations or initializing with default: {e}")
    else:
        st.session_state.users = {}
        st.session_state.all_conversations = {}

def initialize_session_state():
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = "default"
    if 'users' not in st.session_state or 'all_conversations' not in st.session_state:
        load_data()

    if st.session_state.username:
        if 'conversations' not in st.session_state:
            st.session_state.conversations = st.session_state.all_conversations.get(st.session_state.username, {"default": []})

def display_chat(conversation_name):
    for message in st.session_state.conversations.get(conversation_name, []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def save_and_rerun():
    save_data()
    st.experimental_rerun()

def login():
    username = st.session_state["username_input"]
    password = st.session_state["password_input"]

    if username in st.session_state.users and st.session_state.users[username] == password:
        st.session_state.username = username
        if username not in st.session_state.all_conversations:
            st.session_state.all_conversations[username] = {"default": [{"role": "assistant", "content": "Hello! How can I help you today?"}]}
        st.session_state.conversations = st.session_state.all_conversations[username]
        save_and_rerun()
    else:
        st.error("Invalid username or password")

def signup():
    username = st.session_state["username_signup"]
    password = st.session_state["password_signup"]

    if username not in st.session_state.users:
        st.session_state.users[username] = password
        st.session_state.all_conversations[username] = {"default": [{"role": "assistant", "content": "Hello! How can I help you today?"}]}
        st.session_state.username = username
        st.session_state.conversations = st.session_state.all_conversations[username]
        save_and_rerun()
    else:
        st.error("Username already taken")

initialize_session_state()

if 'username' not in st.session_state or not st.session_state.username:
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        with st.form(key="login_form"):
            st.text_input("Username", key="username_input")
            st.text_input("Password", type="password", key="password_input")
            st.form_submit_button("Login", on_click=login)

    with tab2:
        with st.form(key="signup_form"):
            st.text_input("Username", key="username_signup")
            st.text_input("Password", type="password", key="password_signup")
            st.form_submit_button("Sign up", on_click=signup)
else:
    st.title(f"Hello, {st.session_state.username}!")

    with st.sidebar:
        st.header(f"Hello, {st.session_state.username}!", divider="gray")
        st.header("A little project by Wize")

        with st.form(key="new_convo_form"):
            new_convo_name = st.text_input("New Conversation Name")
            create_button = st.form_submit_button("➕")

        if create_button and new_convo_name:
            if new_convo_name not in st.session_state.conversations:
                st.session_state.conversations[new_convo_name] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
            st.session_state.current_conversation = new_convo_name
            save_and_rerun()

        with st.expander("Recent Conversations"):
            for convo in list(st.session_state.conversations.keys()):
                col1, col2 = st.columns([8, 2])
                with col1:
                    if st.button(convo):
                        st.session_state.current_conversation = convo
                        save_and_rerun()
                with col2:
                    if convo != "default" and st.button("🗑️", key=f"delete_{convo}"):
                        del st.session_state.conversations[convo]
                        if st.session_state.current_conversation == convo:
                            st.session_state.current_conversation = "default"
                        save_and_rerun()

        with st.expander("Settings"):
            st.header("Settings")

            available_models = ["gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct", 
                                "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613",
                                "gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613",
                                "gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo-preview",
                                "gpt-4-turbo-2024-04-09", "gpt-4-turbo", "gpt-4o"]
            selected_model = st.selectbox("Choose Model", available_models, index=available_models.index("gpt-4o"), disabled=st.session_state.is_processing)

            temperature = st.slider("Temperature", 0.0, 1.0, 1.0, step=0.1, disabled=st.session_state.is_processing)
            max_tokens = st.slider("Max Tokens", 100, 4096, 4096, step=100, disabled=st.session_state.is_processing)
            top_p = st.slider("Top P", 0.0, 1.0, 1.0, step=0.1, disabled=st.session_state.is_processing)

            system_prompt = st.text_area("System Prompt", "You are a helpful AI assistant.")
            if system_prompt:
                chatbot = Chatbot()
                chatbot.system_prompt = system_prompt

    current_conversation = st.session_state.current_conversation
    display_chat(current_conversation)

    user_input = st.chat_input("Ask me anything...", disabled=st.session_state.is_processing)

    if user_input and not st.session_state.is_processing:
        st.session_state.is_processing = True
        st.session_state.conversations[current_conversation].append({"role": "user", "content": user_input})
        st.session_state.all_conversations[st.session_state.username] = st.session_state.conversations
        save_and_rerun()

    if st.session_state.is_processing:
        chatbot = Chatbot()
        chatbot.model = selected_model
        chatbot.temperature = temperature
        chatbot.max_tokens = max_tokens
        chatbot.top_p = top_p
        chatbot.system_prompt = system_prompt

        last_message = st.session_state.conversations[current_conversation][-1]["content"] if st.session_state.conversations[current_conversation] else ""

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_text = ""
            for chunk in chatbot.send_message(st.session_state.conversations[current_conversation], last_message):
                response_text += chunk
                response_placeholder.markdown(response_text, unsafe_allow_html=True)

        st.session_state.is_processing = False
        st.session_state.all_conversations[st.session_state.username] = st.session_state.conversations
        save_and_rerun()
