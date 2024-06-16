import base64
import os
import datetime
import io
import json
from PIL import Image
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
from PyPDF2 import PdfReader
import docx
import tiktoken
from bs4 import BeautifulSoup
from requests.utils import default_headers
import ffmpeg
import extra_streamlit_components as stx

# Function to encode an image to Base64 format
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Function to extract frames from a video
def extract_frames(video_bytes, num_frames=15):
    probe = ffmpeg.probe(video_bytes)
    duration = float(probe['format']['duration'])
    interval = duration / num_frames

    frames = []
    for i in range(num_frames):
        frame_path = f"frame_{i:03d}.jpg"
        (
            ffmpeg.input(video_bytes, ss=i * interval)
            .output(frame_path, vframes=1, format='image2', vcodec='mjpeg')
            .run(capture_stdout=True, capture_stderr=True)
        )
        frames.append(frame_path)
    return frames

# Function to create a grid image from multiple frames
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

# Class to handle chat functionalities
class Chatbot:
    # Initialize chatbot
    def __init__(self):
        self.api_url = "https://omniplex.ai/api/chat"
        self.model = "gpt-4o"
        self.temperature = 1
        self.max_tokens = 4096
        self.top_p = 1
        self.system_prompt = "You are a helpful AI assistant."

    def send_message(self, chat_history, message, websearch=True):
        if websearch:
            generated_query = generate_search_query(message, chat_history)
            if generated_query != "CANCEL_WEBSEARCH":
                queries = generated_query.strip().split('\n')
                site = st.session_state.get("site_input", "").strip()
                if site:
                    queries = [f"site:{site} {query}" for query in queries]

                scraped_results_json = scrape_and_process_results(queries, 3)

                current_time = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y, %H:%M:%S UTC")
                user_name = st.session_state.username

                message = f"""{message}

Write an accurate answer concisely for a given question in English, always always citing the search results. Your answer must be correct, high-quality, and written by an expert using an unbiased and journalistic tone. Always cite search results for your responses using hyperlinked superscript numbers of the index at the end of sentences when needed, for example "Ice is less dense than water.[¹](https://example.com/source1)" NO SPACE between the last word and the citation. Cite the most relevant results that answer the question. Avoid citing irrelevant results. Write only the response. Use markdown for formatting.

Use markdown to format paragraphs, lists, tables, and quotes whenever possible. 
Use markdown code blocks to write code, including the language for syntax highlighting.
Use LaTeX to wrap ALL math expressions. Always use double dollar signs $$, for example $$E=mc^2$$.
DO NOT include any URL's, only include hyperlinked citations with superscript numbers, e.g. [¹](https://example.com/source1)
DO NOT include references (URL's at the end, sources).
Use hyperlinked footnote citations at the end of applicable sentences (e.g, [¹](https://example.com/source1)[²](https://example.com/source2)).
Write more than 100 words (2 paragraphs).
ALWAYS use the exact cite format provided.
ONLY cite sources from search results below. DO NOT add any other links other than the search results below

{scraped_results_json}
"""

        payload = {
            "messages": [{"role": "system", "content": self.system_prompt}] + chat_history + [{"role": "user", "content": message}],
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
        with st.spinner(""):
            response = requests.post(self.api_url, headers={'Content-Type': 'application/json'}, json=payload, stream=True)
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    chunk_str = chunk.decode("utf-8")
                    response_text += chunk_str
                    yield response_text + typing_indicator
            chat_history.append({"role": "assistant", "content": response_text})
            return chat_history
        else:
            yield f"Error: {response.status_code}"

# Initialization code
INITIAL_MESSAGE = {"role": "assistant", "content": "Hello! How can I help you today?"}
MODEL_CONTEXT_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16384,
}
DATA_FILE = "conversations.json"

# Streamlit page config
st.set_page_config(page_title="Lumiere", page_icon="🐈‍⬛", layout="wide")

# Hide main menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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
    if "users" not in st.session_state or "all_conversations" not in st.session_state:
        users, all_conversations = load_data()
        st.session_state.users = users
        st.session_state.all_conversations = all_conversations

    # Check for cookies
    username_cookie = cookie_manager.get(cookie="username")
    if username_cookie:
        st.session_state.username = username_cookie

    # Initialize the conversations for the logged-in user
    if st.session_state.username:
        if "conversations" not in st.session_state:
            st.session_state.conversations = st.session_state.all_conversations.get(st.session_state.username, {})

        # Set current conversation to the most recent one if not already set or if it doesn't exist
        if "current_conversation" not in st.session_state or not st.session_state.current_conversation:
            if st.session_state.conversations:
                most_recent_convo = sorted(st.session_state.conversations.keys())[-1]
                st.session_state.current_conversation = most_recent_convo
            else:
                st.session_state.current_conversation = None

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
        avatar_url = ":material/person:" if message["role"] == "user" else "https://i.ibb.co/4PbTLG9/20240531-141431.jpg"

        with st.chat_message(message["role"], avatar=avatar_url):
            if "file_contents" in message:
                with st.expander(f"{message['file_name']}"):
                    st.text(message["file_contents"])
            elif "images" in message:
                for image in message["images"]:
                    try:
                        image_data = image.split(",")[1]
                        st.image(base64.b64decode(image_data), caption="Uploaded Image")
                    except IndexError:
                        st.toast("Error displaying image: Invalid base64 string.", icon="❌")
            elif "video" in message:
                st.video(message["video"])
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
            st.session_state.all_conversations[username] = {}
        st.session_state.conversations = st.session_state.all_conversations[username]
        create_new_conversation()
    else:
        st.toast("Invalid username or password", icon="❌")

def signup():
    username = st.session_state["username_signup"]
    password = st.session_state["password_signup"]

    if username not in st.session_state.users:
        st.session_state.users[username] = password

        # Set cookies
        cookie_manager.set("username", username, expires_at=datetime.datetime.now() + datetime.timedelta(days=30))

        st.session_state.all_conversations[username] = {}
        st.session_state.username = username
        st.session_state.conversations = st.session_state.all_conversations[username]
        create_new_conversation()
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
    keys_to_clear = ["username", "current_conversation", "is_processing", "conversations"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    components.html(delete_cookies_script())
    st.stop()
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

def process_video_upload(uploaded_file):
    video_bytes = uploaded_file.read()

    # Display video preview
    st.video(video_bytes)

    # Extract frames and create an image grid
    frames = extract_frames(io.BytesIO(video_bytes), num_frames=15)
    grid_image_path = create_image_grid(frames)

    # Encode the grid image to base64
    grid_image = Image.open(grid_image_path)
    encoded_grid_image = encode_image(grid_image)

    st.session_state.conversations[st.session_state.current_conversation].append({
        "role": "user",
        "content": [{"type": "text", "text": "What’s in this video?"}],
        "images": [f"data:image/jpeg;base64,{encoded_grid_image}"],
        "video": video_bytes
    })
    st.session_state.all_conversations[st.session_state.username] = st.session_state.conversations

    # Clean up frame files
    for frame_path in frames:
        os.remove(frame_path)
    os.remove(grid_image_path)

    st.session_state.is_processing = False
    save_and_rerun()

def handle_file_upload():
    selected_model = st.session_state.selected_model
    model_context_window = MODEL_CONTEXT_LIMITS.get(selected_model, 4096)

    uploaded_files = st.session_state.file_uploader

    if uploaded_files:
        st.session_state.is_processing = True

        encoding = tiktoken.encoding_for_model(selected_model)

        for uploaded_file in uploaded_files:
            file_contents = uploaded_file.read()
            filename = uploaded_file.name.lower()

            valid_text_extensions = [
                ".py", ".txt", ".json", ".js", ".css", ".html", ".xml", ".csv", ".tsv", ".yaml", ".yml", ".ini", ".md", ".log",
                ".bat", ".sh", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rb", ".swift", ".kt", ".kts", ".rs",
                ".pdf", ".docx"
            ]
            valid_image_extensions = ["jpg", "jpeg", "png"]
            valid_video_extensions = ["mp4", "avi", "mov"]

            if any(filename.endswith(ext) for ext in valid_text_extensions):
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
                allowed_models = {"gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-4-turbo-2024-04-09"}
                if selected_model not in allowed_models:
                    st.toast("Image uploads are only supported by models that accept image inputs.", icon="⚠️")
                    st.session_state.is_processing = False
                    return

                file_contents = compress_image(io.BytesIO(file_contents))
                send_image_multipart(uploaded_file.name, file_contents)

                encoded_image = base64.b64encode(file_contents).decode("utf-8")
                mime_type = Image.open(io.BytesIO(file_contents)).format.lower()
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
            elif any(filename.endswith(ext) for ext in valid_video_extensions):
                allowed_models = {"gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-4-turbo-2024-04-09"}
                if selected_model not in allowed_models:
                    st.toast("Video uploads are only supported by models that accept video inputs.", icon="⚠️")
                    st.session_state.is_processing = False
                    return

                # Process the video upload
                process_video_upload(uploaded_file)
            else:
                st.toast("Only text-based files, image files, and video files like .py, .txt, .json, .js, .jpg, .jpeg, .png, .pdf, .docx, .mp4, .avi, .mov are allowed.", icon="❌")

def reset_current_conversation():
    st.session_state.conversations[st.session_state.current_conversation] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
    st.session_state.all_conversations[st.session_state.username] = st.session_state.conversations
    save_and_rerun()

def count_tokens(message, encoding):
    try:
        return len(encoding.encode(message, disallowed_special=()))
    except ValueError as e:
        st.toast(f"Tokenization error: {str(e)}", icon="❌")
        return float("inf")

def generate_search_query(query, chat_history):
    current_time = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y, %H:%M:%S UTC")
    prompt = f"""## Additional Context
- The date and time is {current_time}.
- Do not generate any other texts or messages other than the search queries, do not engage in a conversation. You are not a chatbot, an AI, an assistant or any other form. NEVER EVER SAY ANYTHING ELSE, OTHER THAN THE SEARCH QUERY. YOU ARE NOT A CHATBOT OR AN ASSISTANT. YOU ARE A QUERY GENERATION SYSTEM.

## You are a query generation system. Your knowledge cutoff date is August 2023.

If the input is a factual query or requires up-to-date information, generate 1-3 focused search queries. For casual conversation, thank you messages, or statements, respond with "c". Use formal language and keep the queries concise, avoiding unnecessary details.

Input: can you help me study
Output: c
Input: find me open-source projects
Output: open source projects github
Input: Best ways to save money on groceries
Output: grocery saving tips
Input: What is the weather forecast for tomorrow in San Francisco?
Output: san francisco weather forecast
Input: How can I learn to code in Python?
Output: python programming tutorials for beginners
learn python coding
Input: Thank you for your help!
Output: c
"""

    url = "https://omniplex.ai/api/chat"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "messages": [{"role": "system", "content": prompt}] + chat_history + [{"role": "user", "content": "Generate a search query based on this input, do not engage in a conversation or provide commentary. Do not respond with anything other than the search query. Input: " + query}],
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": False
    }
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        response_text = response.text
        if response_text.lower() == "c":
            return "CANCEL_WEBSEARCH"
        else:
            return response_text
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return "CANCEL_WEBSEARCH"

def omniplex_search(query):
    url = "https://omniplex.ai/api/search"
    params = {
        'q': query
    }
    headers = {
        'User-Agent': "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
    }
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        urls = [result['url'] for result in data['data']['webPages']['value']]
        return urls
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

def omniplex_scrape(urls):
    url = "https://omniplex.ai/api/scrape"
    params = {
        'urls': ",".join(urls)
    }
    headers = {
        'User-Agent': "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
    }
    response = requests.post(url, params=params, headers=headers)

    if response.status_code == 200:
        return response.text
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return ""

def scrape_and_process_results(queries, max_results_per_query):
    all_results_json = []
    num_queries = len(queries)
    if num_queries == 3:
        max_results_per_query = 2
    elif num_queries == 2:
        max_results_per_query = 3
    else:
        max_results_per_query = 4

    for query_idx, query in enumerate(queries):
        urls = omniplex_search(query)
        if urls:
            scraped_data = omniplex_scrape(urls[:max_results_per_query])
            results_json = []
            for idx, url in enumerate(urls[:max_results_per_query], start=1):
                results_json.append({
                    "index": idx + query_idx * max_results_per_query,
                    "url": url,
                    "content": scraped_data
                })
            all_results_json.extend(results_json)
    return json.dumps(all_results_json)

initialize_session_state()

def generate_title(prompt):
    url = "https://omniplex.ai/api/chat"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "messages": [{"role": "system", "content": "Generate a short conversation title based on the user's message. Provide only the title without additional text or commentary."},
                     {"role": "user", "content": prompt}],
        "model": "gpt-4o",
        "temperature": 0.5,
        "max_tokens": 50,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": False
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        title = response.text.strip()
        return title
    else:
        st.toast(f"Error generating title: {response.status_code}", icon="❌")
        return "Untitled"

def create_new_conversation():
    new_chat_name = "New Chat"
    st.session_state.conversations[new_chat_name] = [INITIAL_MESSAGE]
    st.session_state.current_conversation = new_chat_name
    save_and_rerun()

# Main UI function
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
        if not st.session_state.conversations:
            create_new_conversation()
        current_conversation = st.session_state.current_conversation

        if current_conversation not in st.session_state.conversations:
            if st.session_state.conversations:
                st.session_state.current_conversation = sorted(st.session_state.conversations.keys())[-1]
                current_conversation = st.session_state.current_conversation
            else:
                st.session_state.current_conversation = None
                current_conversation = None

        with st.sidebar:
            st.header(f"Welcome to Lumiere, {st.session_state.username}.")

            st.file_uploader("Upload files (text, images, videos)", type=[
                "py", "txt", "json", "js", "css", "html", "xml", "csv", "tsv", "yaml", "yml",
                "ini", "md", "log", "bat", "sh", "java", "c", "cpp", "h", "hpp", "cs", "go",
                "rb", "swift", "kt", "kts", "rs", "jpg", "jpeg", "png", "pdf", "docx", "mp4", "avi", "mov"
            ], key="file_uploader", accept_multiple_files=True, on_change=handle_file_upload)

            create_button = st.button("➕", on_click=create_new_conversation)

            with st.expander("History"):
                for convo in list(st.session_state.conversations.keys()):
                    col1, col2 = st.columns([8, 2])
                    with col1:
                        if st.button(convo):
                            st.session_state.current_conversation = convo
                            save_and_rerun()
                    with col2:
                        if convo != "New Chat" and st.button("🗑️", key=f"delete_{convo}"):
                            del st.session_state.conversations[convo]
                            if st.session_state.current_conversation == convo:
                                st.session_state.current_conversation = None
                                if not st.session_state.conversations:
                                    create_new_conversation()
                            save_and_rerun()

            with st.expander("Settings"):
                websearch = st.checkbox("Web Search", value=True)
                site = st.text_input("Site", placeholder="stackoverflow.com", key="site_input")

                available_models = list(MODEL_CONTEXT_LIMITS.keys())
                selected_model = st.selectbox("Choose Model", available_models, index=available_models.index("gpt-4o"), disabled=st.session_state.is_processing)
                st.session_state.selected_model = selected_model

                temperature = st.slider("Temperature", 0.0, 1.0, 1.0, step=0.1, disabled=st.session_state.is_processing)
                max_tokens = st.slider("Max Tokens", 100, 4096, 4096, step=100, disabled=st.session_state.is_processing)
                top_p = st.slider("Top P", 0.0, 1.0, 1.0, step=0.1, disabled=st.session_state.is_processing)

                system_prompt = st.text_area("System Prompt", "You are a helpful AI assistant.", disabled=st.session_state.is_processing)

                encoding = tiktoken.encoding_for_model(selected_model)

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

                generated_query = generate_search_query(content_to_send, st.session_state.conversations[current_conversation])
                if websearch and generated_query and generated_query != "CANCEL_WEBSEARCH":
                    queries = generated_query.strip().split('\n')
                    site = st.session_state.get("site_input", "").strip()
                    if site:
                        queries = [f"site:{site} {query}" for query in queries]
                    for query in queries:
                        clean_query = query.replace(f"site:{site} ", "").strip()
                        st.write(f"Searching 🌐 for {clean_query}")

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

main_ui()
