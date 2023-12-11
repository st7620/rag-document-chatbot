import os
# Disable parallelism warming
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import streamlit as st
from txtai import Embeddings
import nltk
nltk.download('punkt')
import tempfile
from transformers import pipeline
from pypdf import PdfReader
import io


# App title
st.set_page_config(page_title="Document Chatbot")

# File uploader widget  
uploaded_files = st.file_uploader("Choose your files", accept_multiple_files=True)

# Load embedding model
@st.cache_resource
def load_embeddings():
    return Embeddings(path="sentence-transformers/nli-mpnet-base-v2", content=True, objects=True)

embeddings = load_embeddings()

# Load NLP model
@st.cache_resource
def load_nlp_model():
    return pipeline("question-answering",
                    model="deepset/roberta-base-squad2",
                    tokenizer="deepset/roberta-base-squad2",)

nlp = load_nlp_model()

# Check if there are any uploaded files
if uploaded_files:
    data = []
    for file in uploaded_files:
        file_contents = file.read()
        remote_file_bytes = io.BytesIO(file_contents)
        pdfdoc_remote = PdfReader(remote_file_bytes)

        pdf_text = ""

        # Extract text by page
        for i in range(len(pdfdoc_remote.pages)):
            print(i)
            page = pdfdoc_remote.pages[i]
            page_content = page.extract_text()
            pdf_text += page_content

        # Add text to data list
        data.append(pdf_text)

    # Create an index for the list of text
    embeddings.index(data)

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating response
def generate_response(prompt_input):                       
    context = embeddings.search(prompt_input, 1)[0]["text"]
    question_set = {"context": context, "question": prompt_input}
    return nlp(question_set)["answer"]

# User-provided prompt
if prompt := st.chat_input():
    print("user prompt: ")
    print(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        print("trying to write")
        st.write(prompt)
        print("after write")

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            print("responding to prompt: ")
            print(prompt)
            print("response:")
            response = generate_response(prompt) 
            print(response)
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)