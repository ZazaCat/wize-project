import base64
import requests
import json
import ffmpeg
import os
from PIL import Image
import streamlit as st

INITIAL_MESSAGE = {"role": "assistant", "content": "Hello! How can I help you today?"}
MODEL_CONTEXT_LIMITS = {"gpt-4o": 128000, "gpt-4-turbo-2024-04-09": 128000, "gpt-4-turbo": 128000, "gpt-4": 8192, "gpt-3.5-turbo": 16384}
DATA_FILE = "conversations.json"

# Constants for Omniplex API URL, model, and headers
API_URL = "https://omniplex.ai/api/chat"
MODEL = "gpt-4o"
HEADERS = {'Content-Type': "application/json"}
VIDEO_FOLDER = "uploaded_videos"
FRAME_FOLDER = "extracted_frames"

@st.cache_resource
def get_video(path: str):
    return open(path, "rb").read()

def extract_frames(video_bytes, num_frames=15):
    if not os.path.exists(FRAME_FOLDER):
        os.makedirs(FRAME_FOLDER)
    video_path = os.path.join(VIDEO_FOLDER, "temp_video.mp4")
    with open(video_path, "wb") as f:
        f.write(video_bytes)
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    interval = duration / num_frames
    frames = []
    for i in range(num_frames):
        frame_path = os.path.join(FRAME_FOLDER, f"frame_{i:03d}.jpg")
        (
            ffmpeg.input(video_path, ss=i * interval)
            .output(frame_path, vframes=1, format='image2', vcodec='mjpeg')
            .run(capture_stdout=True, capture_stderr=True)
        )
        frames.append(frame_path)
    os.remove(video_path)
    return frames

def create_image_grid(image_paths, grid_size=(5, 3), output_path="grid_image.jpg"):
    images = [Image.open(img_path) for img_path in image_paths]
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)
    grid_width = max_width * grid_size[0]
    grid_height = max_height * grid_size[1]
    grid_img = Image.new('RGB', (grid_width, grid_height))
    for idx, img in enumerate(images):
        row = idx // grid_size[0]
        col = idx % grid_size[0]
        grid_img.paste(img, (col * max_width, row * max_height))
    grid_img.save(output_path, quality=95)
    return output_path

def encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_and_rerun():
    save_data(st.session_state.users, st.session_state.all_conversations)
    raise RerunException(RerunData())

def initialize_session_state():
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "users" not in st.session_state or "all_conversations" not in st.session_state:
        users, all_conversations = load_data()
        st.session_state.users = users
        st.session_state.all_conversations = all_conversations

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

def show_uploaded_video():
    uploaded_files = st.session_state.file_uploader

    if uploaded_files:
        st.session_state.is_processing = True

        for uploaded_file in uploaded_files:
            file_contents = uploaded_file.read()
            filename = uploaded_file.name.lower()

            valid_video_extensions = [".mp4", ".mov", ".avi"]

            if any(filename.endswith(ext) for ext in valid_video_extensions):
                st.video(file_contents)
                frames = extract_frames(file_contents)
                grid_image_path = create_image_grid(frames)

                encoded_grid_image = encoded_image(grid_image_path)

                st.session_state.conversations[st.session_state.current_conversation].append({
                    "role": "user",
                    "content": [{"type": "text", "text": "What’s in this video?"}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_grid_image}"}]
                })
                st.session_state.all_conversations[st.session_state.username] = st.session_state.conversations

                st.session_state.is_processing = False
                save_and_rerun()
            else:
                st.toast("Only video-based files like .mp4, .mov, .avi are allowed.", icon="❌")

def save_and_rerun():
    save_data(st.session_state.users, st.session_state.all_conversations)
    st.rerun()

def main_ui():
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
        current_conversation = st.session_state.current_conversation
        if not st.session_state.conversations:
            create_new_conversation()
        if current_conversation not in st.session_state.conversations:
            if st.session_state.conversations:
                st.session_state.current_conversation = sorted(st.session_state.conversations.keys())[-1]
                current_conversation = st.session_state.current_conversation
            else:
                st.session_state.current_conversation = None
                current_conversation = None

        with st.sidebar:
            st.header(f"Welcome to Lumiere, {st.session_state.username}.")

            st.file_uploader("Upload files (text, images, or videos)", type=[
                "py", "txt", "json", "js", "css", "html", "xml", "csv", "tsv", "yaml", "yml",
                "ini", "md", "log", "bat", "sh", "java", "c", "cpp", "h", "hpp", "cs", "go",
                "rb", "swift", "kt", "kts", "rs", "jpg", "jpeg", "png", "pdf", "docx", "mp4", "mov", "avi"
            ], key="file_uploader", accept_multiple_files=True, on_change=show_uploaded_video)

            create_button = st.button("➕", on_click=create_new_conversation)

            with st.expander("History"):
                for convo in list(st.session_state.conversations.keys()):
                    col1, col2 = st.columns([8, 2])
                    with col1:
                        if st.button(convo):
                            st.session_state.current_conversation = convo
                            raise RerunException(RerunData())
                    with col2:
                        if convo != "New Chat" and st.button("🗑️", key=f"delete_{convo}"):
                            del st.session_state.conversations[convo]
                            if st.session_state.current_conversation == convo:
                                st.session_state.current_conversation = None
                                if not st.session_state.conversations:
                                    create_new_conversation()
                            raise RerunException(RerunData())

            st.button("Logout", on_click=logout)
        if current_conversation:
            st.title(f"{current_conversation or 'No Chat Selected'}")
            display_chat(current_conversation)
            user_input = st.chat_input("Send a message", disabled=st.session_state.is_processing)
            if user_input and not st.session_state.is_processing:
                model_context_window = MODEL_CONTEXT_LIMITS.get(selected_model, 4096)
                user_input_tokens = count_tokens(user_input, encoding)
                if user_input_tokens > model_context_window:
                    st.toast("Your message is too long.", icon="❌")
                else:
                    st.session_state.is_processing = True
                    if current_conversation == "New Chat":
                        title = generate_title(user_input)
                        st.session_state.conversations[title] = st.session_state.conversations.pop(current_conversation)
                        current_conversation = title
                        st.session_state.current_conversation = title
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

                if st.session_state.conversations[current_conversation]:
                    last_message = st.session_state.conversations[current_conversation][-1]
                    content_to_send = last_message.get("file_contents", last_message["content"])
                else:
                    content_to_send = ""
                with st.chat_message("assistant", avatar="https://i.ibb.co/4PbTLG9/20240531-141431.jpg"):
                    response_placeholder = st.empty()
                    response_text = ""
                    typing_indicator = " |"
                    for chunk in chatbot.send_message(st.session_state.conversations[current_conversation], content_to_send, websearch=websearch):
                        response_text = chunk.replace(typing_indicator, "")
                        response_placeholder.markdown(response_text + typing_indicator, unsafe_allow_html=True)
                    response_placeholder.markdown(response_text, unsafe_allow_html=True)
                st.session_state.is_processing = False
                st.session_state.all_conversations[st.session_state.username] = st.session_state.conversations
                save_and_rerun()

def login():
    username = st.session_state["username_input"]
    password = st.session_state["password_input"]
    if username in st.session_state.users and st.session_state.users[username] == password:
        st.session_state.username = username
        if username not in st.session_state.all_conversations:
            st.session_state.all_conversations[username] = {}
        st.session_state.conversations = st.session_state.all_conversations[username]
        create_new_conversation()
    else:
        st.toast("Invalid username or password", icon="❌")

def signup():
    if st.session_state["username_signup"] not in st.session_state.users:
        st.session_state.users = st.session_state["username_signup"]
        st.session_state.conversations = {}
        st.session_state.all_conversations[st.session_state["username_signup"]] = {}
        st.session_state.username = st.session_state["username_signup"]
        create_new_conversation()
    else:
        st.toast("Username already taken", icon="❌")

def logout():
    keys_to_clear = ["username", "current_conversation", "is_processing", "conversations"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

def create_new_conversation():
    new_chat_name = "New Chat"
    st.session_state.conversations[new_chat_name] = [INITIAL_MESSAGE]
    st.session_state.current_conversation = new_chat_name
    save_and_rerun()

initialize_session_state()

if __name__ == "__main__":
    main_ui()
