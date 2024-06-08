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

INITIAL_MESSAGE = {"role": "assistant", "content": "Hello! How can I help you today?"}

# Define context window limits for models
MODEL_CONTEXT_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16384,
}

# Streamlit page config
st.set_page_config(
    page_title="Lumiere",
    page_icon="🐈‍⬛",
    layout="wide",
)

# Hide main menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Custom CSS to change link appearance to a white filled box with a black number index
custom_css = """
    <style>
    a {
        background-color: white;
        color: black !important;
        display: inline-block;
        border-radius: 5px;
        padding: 2px 5px; /* Adjust padding */
        text-decoration: none;
        position: relative;
        top: -2px; /* Adjust based on your needs */
        font-size: 80%; /* Scale down the text size if necessary */
    }
    a:hover {
        background-color: #f0f0f0;
    }
    </style>
"""

# Embed the custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

class Chatbot:
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
                scraped_results_json = scrape_and_process_results(queries, 3)  # Scrape top 3 results for each query

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
                    yield response_text + typing_indicator  # Include typing indicator
            chat_history.append({"role": "assistant", "content": response_text})
            return chat_history
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
            st.session_state.all_conversations[username] = {}
        st.session_state.conversations = st.session_state.all_conversations[username]
        create_new_conversation()  # Create new conversation upon login
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
        create_new_conversation()  # Create new conversation upon signup
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

def handle_file_upload():
    selected_model = st.session_state.selected_model
    model_context_window = MODEL_CONTEXT_LIMITS.get(selected_model, 4096)

    uploaded_files = st.session_state.file_uploader

    if uploaded_files:
        st.session_state.is_processing = True

        # Define encoding using tiktoken
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

                # Compress image if necessary
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
            else:
                st.toast("Only text-based files and image files like .py, .txt, .json, .js, .jpg, .jpeg, .png, .pdf, .docx etc. are allowed.", icon="❌")

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

def generate_search_query(query, chat_history):
    current_time = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y, %H:%M:%S UTC")
    prompt = f"""## Additional Context
- The date and time is {current_time}.
- Do not generate any other texts or messages other than the search queries, do not engage in a conversation. You are not a chatbot, an AI, an assistant or any other form. (IMPORTANT). NEVER EVER SAY ANYTHING ELSE, OTHER THAN THE SEARCH QUERY. YOU ARE NOT A CHATBOT OR AN ASSISTANT. YOU ARE A QUERY GENERATION SYSTEM.

## You are a query generation system designed solely to provide relevant search queries based on the user's input. If the input does not require up-to-date information or refers to something beyond your knowledge base, you may generate 1-3 focused search queries separated by newlines when needed. Your knowledge cutoff date is August 2023.

## If the input is a casual conversation, a statement, an opinion, a thank you message, or any other input that does not require a factual search query response, you must respond with "c". Do not attempt to engage in conversation or provide any other responses.

## However, if the input is a factual query requiring the most current information, data that may have changed since August 2023, or something unfamiliar to you, you may generate 1-3 concise and focused Google search queries separated by newlines to retrieve relevant up-to-date information when a single query is insufficient.

## Generate multiple queries only if the question is complex and broad, requiring additional context or perspectives for a comprehensive answer. For simple factual queries, a single focused query should suffice.

## When generating search queries, strictly adhere to these guidelines:

- Keep it Simple:
Google search is intelligent, so you don't need to be overly specific. For example, to find nearby pizza places, use: "Pizza places near me"

- Use Professional WebsiteTerminology:
Websites often use formal language, unlike casual speech. For better results, use terms found on professional websites. For example:
- Instead of "I have a flat tire", use "repair a flat tire".
- Instead of "My head hurts", use "headache remedies".

- Use Important Keywords Only:
Google search is intelligent, so you don't need to be overly specific. Using too many words may limit results and make it harder to find what you need. Use only the most important keywords when searching. For example:
- Don't use: "Where can I find a Chinese restaurant that delivers?"
- Instead, try: "Chinese takeout near me"

- Use Descriptive Words:
Things can be described in multiple ways. If you struggle to find what you're searching for, rephrase the query using different descriptive words. For example, instead of "How to install drivers in Ubuntu?", try "Ubuntu driver installation troubleshooting".

## You are strictly a query generation system. You do not engage in conversation or provide any other responses besides outputting focused search queries or "c". You have no additional capabilities.

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
Input: Tell me a joke.
Output: c
Input: What are some healthy dinner recipes?
Output: healthy dinner recipes\neasy healthy meals
Input: I want to buy a new laptop, what are the best options in 2024?
Output: best laptops 2024\nlaptop reviews 2024
Input: Can you recommend a good plumber in Chicago?
Output: chicago plumbers\nplumbing services chicago
Input: How do I install the latest graphics drivers for my NVIDIA GPU on Ubuntu 22.04?
Output: install nvidia graphics drivers ubuntu 22.04\nubuntu 22.04 nvidia driver installation"""

    # Send prompt to Omniplex
    url = "https://omniplex.ai/api/chat"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "messages": [{"role": "system", "content": prompt}] + chat_history + [{"role": "user", "content": "Generate a search query based on this input, do not engage in a conversation, no commentary, you are not a conversational chatbot or an assistant. Do not say anything else, or respond with other messages other than the search query, If the input is a casual conversation, a statement, an opinion, a thank you message, or any other input that does not require a factual search query response, you must respond only with 'c'. Do not attempt to engage in conversation or provide any other responses. Here's my input: " + query}],
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
        'authority': "omniplex.ai",
        'accept-language': "en-PH,en-US;q=0.9,en;q=0.8",
        'referer': "https://omniplex.ai/chat/Wk3rQUxprd",
        'sec-ch-ua': "\"Not-A.Brand\";v=\"99\", \"Chromium\";v=\"124\"",
        'sec-ch-ua-mobile': "?1",
        'sec-ch-ua-platform': "\"Android\"",
        'sec-fetch-dest': "empty",
        'sec-fetch-mode': "cors",
        'sec-fetch-site': "same-origin",
        'Cookie': "_ga=GA1.1.1059292211.1715883464; _clck=fw8ekm%7C2%7Cflx%7C0%7C1597; _ga_4L0TGM4R80=GS1.1.1716182640.5.1.1716184046.0.0.0; _clsk=z2swc6%7C1716184049711%7C8%7C1%7Cu.clarity.ms%2Fcollect"
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
        'authority': "omniplex.ai",
        'accept-language': "en-PH,en-US;q=0.9,en;q=0.8",
        'content-type': "application/json",
        'origin': "https://omniplex.ai",
        'referer': "https://omniplex.ai/chat/Wk3rQUxprd",
        'sec-ch-ua': "\"Not-A.Brand\";v=\"99\", \"Chromium\";v=\"124\"",
        'sec-ch-ua-mobile': "?1",
        'sec-ch-ua-platform': "\"Android\"",
        'sec-fetch-dest': "empty",
        'sec-fetch-mode': "cors",
        'sec-fetch-site': "same-origin",
        'Cookie': "_ga=GA1.1.1059292211.1715883464; _clck=fw8ekm%7C2%7Cflx%7C0%7C1597; _ga_4L0TGM4R80=GS1.1.1716182640.5.1.1716184046.0.0.0; _clsk=z2swc6%7C1716184064563%7C9%7C1%7Cu.clarity.ms%2Fcollect"
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
        max_results_per_query = 2  # Top 1 result for each query if there are 3 queries
    elif num_queries == 2:
        max_results_per_query = 3  # Top 2 results for each query if there are 2 queries
    else:
        max_results_per_query = 4  # Top 3 results for single query

    for query_idx, query in enumerate(queries):
        urls = omniplex_search(query)
        if urls:
            scraped_data = omniplex_scrape(urls[:max_results_per_query])
            results_json = []
            for idx, url in enumerate(urls[:max_results_per_query], start=1):
                results_json.append({
                    "index": idx + query_idx * max_results_per_query,
                    "url": url,
                    "content": scraped_data  # Assuming omniplex_scrape returns text for each URL
                })
            all_results_json.extend(results_json)  # Use extend to combine lists
    return json.dumps(all_results_json)

initialize_session_state()

def generate_title(prompt):
    url = "https://omniplex.ai/api/chat"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "messages": [{"role": "system", "content": "Generate a very short conversation title based on the user's message. Provide only the title without any additional text or commentary. Do not say anything else or add any other texts. Only the generated title without any quotation marks."},
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
        return title  # Return the title directly
    else:
        st.toast(f"Error generating title: {response.status_code}", icon="❌")
        return "Untitled"

def create_new_conversation():
    new_chat_name = "New Chat"
    st.session_state.conversations[new_chat_name] = [INITIAL_MESSAGE]
    st.session_state.current_conversation = new_chat_name
    save_and_rerun()

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
        # Automatically create a "New Chat" if there are no existing conversations
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

        # Sidebar
        with st.sidebar:
            st.header(f"Welcome to Lumiere, {st.session_state.username}.")

            # File upload
            st.file_uploader("Upload files (text or images, max 4)", type=[
                "py", "txt", "json", "js", "css", "html", "xml", "csv", "tsv", "yaml", "yml",
                "ini", "md", "log", "bat", "sh", "java", "c", "cpp", "h", "hpp", "cs", "go",
                "rb", "swift", "kt", "kts", "rs", "jpg", "jpeg", "png", "pdf", "docx"
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
                st.header("Settings")

                websearch = st.checkbox("Web Search", value=True)

                available_models = list(MODEL_CONTEXT_LIMITS.keys())
                selected_model = st.selectbox("Choose Model", available_models, index=available_models.index("gpt-4o"), disabled=st.session_state.is_processing)
                st.session_state.selected_model = selected_model  # Store the selected model in session state

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

                    # Generate title for the first user message in a new chat
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
                    for query in queries:
                        st.write(f"Searching for {query}")

                with st.chat_message("assistant", avatar="https://i.ibb.co/4PbTLG9/20240531-141431.jpg"):
                    response_placeholder = st.empty()
                    response_text = ""
                    typing_indicator = " |"

                    for chunk in chatbot.send_message(st.session_state.conversations[current_conversation], content_to_send, websearch=websearch):
                        response_text = chunk.replace(typing_indicator, "")  # Temporary display without indicator
                        response_placeholder.markdown(response_text + typing_indicator, unsafe_allow_html=True)

                    response_placeholder.markdown(response_text, unsafe_allow_html=True)

                st.session_state.is_processing = False
                st.session_state.all_conversations[st.session_state.username] = st.session_state.conversations
                save_and_rerun()

# Run the main UI
main_ui()
