import requests
import json
import os
import datetime
import base64
import streamlit as st
from PIL import Image
import io
import numpy as np
import extra_streamlit_components as stx
import streamlit.components.v1 as components
from requests_toolbelt.multipart.encoder import MultipartEncoder
from PyPDF2 import PdfReader
import docx
import tiktoken
from bs4 import BeautifulSoup
from requests.utils import default_headers

# Define context window limits for models
MODEL_CONTEXT_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0613": 32768,
    "gpt-3.5-turbo": 16384,
    "gpt-3.5-turbo-1106": 16384,
    "gpt-3.5-turbo-instruct": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k-0613": 16384,
    # Add other models as necessary...
}

# Streamlit page config
st.set_page_config(
    page_title="Lumiere",
    page_icon="🐈‍⬛",
    layout="wide",
)

class Chatbot:
    def __init__(self):
        self.api_url = "https://omniplex.ai/api/chat"
        self.model = "gpt-4o"
        self.temperature = 1
        self.max_tokens = 4096
        self.top_p = 1
        self.system_prompt = "You are a helpful AI assistant."

    def send_message(self, chat_history, message):
        payload = {
            "messages": [{"role": "user", "content": self.system_prompt}] + chat_history,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": True
        }
        response_text = ""
        typing_indicator = " |"
        with st.spinner("Thinking..."):
            response = requests.post(self.api_url, headers={'Content-Type': 'application/json'}, json=payload, stream=True)
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    chunk_str = chunk.decode("utf-8")
                    response_text += chunk_str
                    yield response_text + typing_indicator  # Include typing indicator
            chat_history.append({"role": "assistant", "content": response_text})
        else:
            yield f"Error: {response.status_code}"

# File path for storing the conversation data
DATA_FILE = "conversations.json"

def get_manager():
    return stx.CookieManager()

cookie_manager = get_manager()

def save_data(users, all_conversations):
    try:
        with open(DATA_FILE, "w") as file:
            json.dump({"users": users, "conversations": all_conversations}, file)
    except IOError as e:
        st.toast(f"Failed to save data: {e}", icon="❌")

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as file:
                data = json.load(file)
                if isinstance(data, dict):
                    return data.get("users", {}), data.get("conversations", {})
                else:
                    raise json.JSONDecodeError("Not a dictionary", "", 0)
        except (json.JSONDecodeError, IOError) as e:
            st.toast(f"Error loading conversations or initializing with default: {e}", icon="❌")
            return {}, {}
    else:
        return {}, {}

def initialize_session_state():
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = "default"
    if "users" not in st.session_state or "all_conversations" not in st.session_state:
        users, all_conversations = load_data()
        st.session_state.users = users
        st.session_state.all_conversations = all_conversations

    # Check for cookies
    username_cookie = cookie_manager.get(cookie="username")
    if username_cookie:
        st.session_state.username = username_cookie

    if st.session_state.username:
        if "conversations" not in st.session_state:
            st.session_state.conversations = st.session_state.all_conversations.get(st.session_state.username, {"default": []})

def extract_text_from_pdf(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text()  # Use the extract_text method from PdfReader
    return text

def extract_text_from_docx(docx_bytes):
    doc = docx.Document(io.BytesIO(docx_bytes))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

def extract_text_from_url(url):
    headers = default_headers()
    headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
    })

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    html_content = response.text
    soup = BeautifulSoup(html_content, "lxml")
    relevant_text = ' '.join([tag.get_text() for tag in soup.find_all(['p', 'h1', 'h2'])])
    return relevant_text

def display_chat(conversation_name):
    for message in st.session_state.conversations.get(conversation_name, []):
        with st.chat_message(message["role"]):
            if "file_contents" in message:
                with st.expander(f"{message['file_name']}"):
                    st.text(message["file_contents"])
            elif "images" in message:
                for image in message["images"]:
                    # Safely decode and display the base64 image
                    try:
                        image_data = image.split(",")[1]
                        st.image(base64.b64decode(image_data), caption="Uploaded Image")
                    except IndexError:
                        st.toast("Error displaying image: Invalid base64 string.", icon="❌")
            else:
                st.markdown(message["content"])

def save_and_rerun():
    save_data(st.session_state.users, st.session_state.all_conversations)
    st.rerun()

def login():
    username = st.session_state["username_input"]
    password = st.session_state["password_input"]

    if username in st.session_state.users and st.session_state.users[username] == password:
        st.session_state.username = username

        # Set cookies
        cookie_manager.set("username", username, expires_at=datetime.datetime.now() + datetime.timedelta(days=30))

        if username not in st.session_state.all_conversations:
            st.session_state.all_conversations[username] = {"default": [{"role": "assistant", "content": "Hello! How can I help you today?"}]}
        st.session_state.conversations = st.session_state.all_conversations[username]
        save_and_rerun()
    else:
        st.toast("Invalid username or password", icon="❌")

def signup():
    username = st.session_state["username_signup"]
    password = st.session_state["password_signup"]

    if username not in st.session_state.users:
        st.session_state.users[username] = password

        # Set cookies
        cookie_manager.set("username", username, expires_at=datetime.datetime.now() + datetime.timedelta(days=30))

        st.session_state.all_conversations[username] = {"default": [{"role": "assistant", "content": "Hello! How can I help you today?"}]}
        st.session_state.username = username
        st.session_state.conversations = st.session_state.all_conversations[username]
        save_and_rerun()
    else:
        st.toast("Username already taken", icon="❌")

def delete_cookies_script():
    return """
    <script>
    document.cookie.split(";").forEach(function(c) { 
        document.cookie = c.trim().split("=")[0] + 
        '=;expires=Thu, 01 Jan 1970 00:00:01 GMT;path=/';
    });
    location.reload();
    </script>
    """

def logout():
    # Clear session state related to user information and chat
    keys_to_clear = ["username", "current_conversation", "is_processing", "conversations"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Attempt to delete the cookie using a script
    components.html(delete_cookies_script())

    # Optional: Stop further execution to allow page reload
    st.stop()

    # Debug logging
    st.write("Logging out: session state and cookies should be cleared now.")
    st.rerun()

def compress_image(image, max_size=1 * 1024 * 1024):
    img = Image.open(image)
    quality = 95
    buffered = io.BytesIO()
    img_format = img.format or "JPEG"
    while True:
        buffered.seek(0)
        img.save(buffered, format=img_format, quality=quality)
        size = buffered.getbuffer().nbytes
        if size <= max_size or quality == 10:
            break
        quality -= 5
    return buffered.getvalue()

def resize_image(image):
    img = Image.open(image)
    img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    # Ensure the format is valid
    img_format = img.format if img.format in ["JPEG", "PNG"] else "JPEG"
    img.save(buffer, format=img_format)
    buffer.seek(0)
    return buffer

def send_image_multipart(file_name, file_content):
    m = MultipartEncoder(
        fields={
            "file": (file_name, file_content, "image/jpeg")
        }
    )

    response = requests.post("https://omniplex.ai/api/chat", data=m, headers={"Content-Type": m.content_type})
    if response.status_code == 200:
        st.toast("Image uploaded successfully", icon="✅")
    else:
        st.toast("Image uploaded", icon="✅")

def handle_file_upload():
    selected_model = st.session_state.selected_model
    model_context_window = MODEL_CONTEXT_LIMITS.get(selected_model, 4096)

    uploaded_files = st.session_state.file_uploader

    if uploaded_files:
        st.session_state.is_processing = True
        for uploaded_file in uploaded_files:
            file_contents = uploaded_file.read()
            filename = uploaded_file.name.lower()

            valid_text_extensions = [
                ".py", ".txt", ".json", ".js", ".css", ".html", ".xml", ".csv", ".tsv", ".yaml", ".yml", ".ini", ".md", ".log",
                ".bat", ".sh", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rb", ".swift", ".kt", ".kts", ".rs",
                ".pdf", ".docx"
            ]
            valid_image_extensions = ["jpg", "jpeg", "png"]

            if any(filename.endswith(ext) for ext in valid_text_extensions):
                # Extract text based on file type
                if filename.endswith(".pdf"):
                    extracted_text = extract_text_from_pdf(file_contents)
                elif filename.endswith(".docx"):
                    extracted_text = extract_text_from_docx(file_contents)
                else:
                    extracted_text = file_contents.decode("utf-8")

                try:
                    tokens = len(encoding.encode(extracted_text, disallowed_special=()))
                except ValueError as e:
                    st.toast(f"Error processing file {uploaded_file.name}: {str(e)}", icon="❌")
                    st.session_state.is_processing = False
                    return

                if tokens > model_context_window:
                    st.toast(f"File {uploaded_file.name} exceeds the token limit of {model_context_window} tokens.", icon="❌")
                    st.session_state.is_processing = False
                    return

                st.session_state.conversations[st.session_state.current_conversation].append({
                    "role": "user",
                    "content": f"File content of {uploaded_file.name}: {extracted_text}",
                    "file_contents": extracted_text,
                    "file_name": uploaded_file.name
                })
                st.session_state.all_conversations[st.session_state.username] = st.session_state.conversations

                st.session_state.is_processing = False
                save_and_rerun()

            elif any(filename.endswith(ext) for ext in valid_image_extensions):
                # Only allow image uploads for models that support image input
                allowed_models = {"gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-4-turbo-2024-04-09"}
                if selected_model not in allowed_models:
                    st.toast("Image uploads are only supported by models that accept image inputs.", icon="⚠️")
                    st.session_state.is_processing = False
                    return

                # Correcting the error in this line
                img = Image.open(io.BytesIO(file_contents))
                if img.width > 1028 or img.height > 1028 or len(file_contents) > 1 * 1024 * 1024:
                    file_contents = compress_image(io.BytesIO(file_contents))
                    resized_image = resize_image(io.BytesIO(file_contents))
                else:
                    resized_image = io.BytesIO(file_contents)

                send_image_multipart(uploaded_file.name, resized_image.getvalue())

                encoded_image = base64.b64encode(resized_image.getvalue()).decode("utf-8")
                mime_type = Image.open(resized_image).format.lower()
                image_base64 = f"data:image/{mime_type};base64,{encoded_image}"

                if "image_bundles" not in st.session_state:
                    st.session_state.image_bundles = []
                st.session_state.image_bundles.append(image_base64)

                if len(st.session_state.image_bundles) == 4 or len(uploaded_files) == st.session_state.file_uploader.index(uploaded_file) + 1:
                    st.session_state.conversations[st.session_state.current_conversation].append({
                        "role": "user",
                        "content": [{"type": "text", "text": "What’s in these images?"}] + [
                            {"type": "image_url", "image_url": {"url": img}} for img in st.session_state.image_bundles
                        ],
                        "images": st.session_state.image_bundles
                    })
                    st.session_state.image_bundles = []
                    st.session_state.all_conversations[st.session_state.username] = st.session_state.conversations

                    st.session_state.is_processing = False
                    save_and_rerun()
            else:
                st.toast("Only text-based files and image files like .py, .txt, .json, .js, .jpg, .jpeg, .png, .pdf, .docx etc. are allowed.", icon="❌")

def handle_article_url():
    selected_model = st.session_state.selected_model
    model_context_window = MODEL_CONTEXT_LIMITS.get(selected_model, 4096)

    article_url = st.session_state.article_url_input.strip()

    if article_url:
        st.session_state.is_processing = True

        try:
            extracted_text = extract_text_from_url(article_url)
            tokens = len(encoding.encode(extracted_text, disallowed_special=()))

            if tokens > model_context_window:
                st.toast(f"Article content exceeds the token limit of {model_context_window} tokens.", icon="❌")
            else:
                st.session_state.conversations[st.session_state.current_conversation].append({
                    "role": "user",
                    "content": f"Extracted content from {article_url}:\n{extracted_text}",
                    "file_contents": extracted_text,
                    "file_name": article_url
                })
                st.session_state.all_conversations[st.session_state.username] = st.session_state.conversations

            st.session_state.is_processing = False
            save_and_rerun()
        except Exception as e:
            st.toast(f"Error extracting text from URL: {str(e)}", icon="❌")
            st.session_state.is_processing = False

def reset_current_conversation():
    st.session_state.conversations[st.session_state.current_conversation] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
    st.session_state.all_conversations[st.session_state.username] = st.session_state.conversations
    save_and_rerun()

def count_tokens(message, encoding):
    # Handle special tokens by allowing them
    try:
        return len(encoding.encode(message, disallowed_special=()))
    except ValueError as e:
        st.toast(f"Tokenization error: {str(e)}", icon="❌")
        return float("inf")  # Return a large number to prevent processing the message

initialize_session_state()

# Main UI
if st.session_state.username is None:
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
    st.title(f"{st.session_state.current_conversation}")

    # Sidebar
    with st.sidebar:
        st.header(f"Welcome to Lumiere, {st.session_state.username}.")

        # File upload
        st.file_uploader("Upload files (text or images, max 4)", type=[
            "py", "txt", "json", "js", "css", "html", "xml", "csv", "tsv", "yaml", "yml",
            "ini", "md", "log", "bat", "sh", "java", "c", "cpp", "h", "hpp", "cs", "go",
            "rb", "swift", "kt", "kts", "rs", "jpg", "jpeg", "png", "pdf", "docx"
        ], key="file_uploader", accept_multiple_files=True, on_change=handle_file_upload)

        st.write("## Article")
        with st.form(key="article_form"):
            st.text_input("Enter URL", key="article_url_input")
            st.form_submit_button("Extract", on_click=handle_article_url)

        with st.form(key="new_convo_form"):
            new_convo_name = st.text_input("New Conversation Name")
            create_button = st.form_submit_button("➕")

            reset_button = st.form_submit_button("🔄", on_click=reset_current_conversation)

        if create_button and new_convo_name:
            if new_convo_name not in st.session_state.conversations:
                st.session_state.conversations[new_convo_name] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
            st.session_state.current_conversation = new_convo_name
            save_and_rerun()

        with st.expander("Conversations"):
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

            available_models = list(MODEL_CONTEXT_LIMITS.keys())
            selected_model = st.selectbox("Choose Model", available_models, index=available_models.index("gpt-4o"), disabled=st.session_state.is_processing)
            st.session_state.selected_model = selected_model  # Store the selected model in session state

            temperature = st.slider("Temperature", 0.0, 1.0, 1.0, step=0.1, disabled=st.session_state.is_processing)
            max_tokens = st.slider("Max Tokens", 100, 4096, 4096, step=100, disabled=st.session_state.is_processing)
            top_p = st.slider("Top P", 0.0, 1.0, 1.0, step=0.1, disabled=st.session_state.is_processing)

            system_prompt = st.text_area("System Prompt", "You are a helpful AI assistant.", disabled=st.session_state.is_processing)

            # Token encoding for the selected model
            encoding = tiktoken.encoding_for_model(selected_model)

        st.button("Logout", on_click=logout)
        st.markdown(f'<a href="https://ko-fi.com/wwize" style="color:#ffffff;font-size:16px;font-family:monospace;">Support Me</a>', unsafe_allow_html=True)

    current_conversation = st.session_state.current_conversation
    display_chat(current_conversation)

    user_input = st.chat_input("Ask me anything...", disabled=st.session_state.is_processing)

    if user_input and not st.session_state.is_processing:
        # Model-specific context window limit
        model_context_window = MODEL_CONTEXT_LIMITS.get(selected_model, 4096)

        # Check token count before proceeding
        user_input_tokens = count_tokens(user_input, encoding)
        if user_input_tokens > model_context_window:
            st.toast("Your message is too long.", icon="❌")
        else:
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

        # Fetch the last message or the file content if available
        if st.session_state.conversations[current_conversation]:
            last_message = st.session_state.conversations[current_conversation][-1]
            content_to_send = last_message.get("file_contents", last_message["content"])
        else:
            content_to_send = ""

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_text = ""
            typing_indicator = ""
            for chunk in chatbot.send_message(st.session_state.conversations[current_conversation], content_to_send):
                response_text = chunk.replace(typing_indicator, "")  # Temporary display without indicator
                response_placeholder.markdown(response_text + typing_indicator, unsafe_allow_html=True)
            # Final response without typing indicator
            response_placeholder.markdown(response_text, unsafe_allow_html=True)

        st.session_state.is_processing = False
        st.session_state.all_conversations[st.session_state.username] = st.session_state.conversations
        save_and_rerun()
