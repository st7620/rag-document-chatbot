import streamlit as st
from txtai import Embeddings
import nltk
nltk.download('punkt')
from txtai.pipeline import Textractor
import tempfile
import os
from transformers import pipeline

# Disable parallelism warming
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# App title
st.set_page_config(page_title="Document Chatbot")

# File uploader widget  
uploaded_files = st.file_uploader("Choose your files", accept_multiple_files=True)

# Check if there are any uploaded files
if uploaded_files:
    data = []
    for file in uploaded_files:
        # Create textractor model
        textractor = Textractor()

        # Get file path
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getvalue())
        
        # Extract text from pdf file
        data.append(textractor(path))

    # Set up model pipeline
    nlp = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2",
    )

    # Create embeddings with content enabled. The default behavior is to only store indexed vectors.
    embeddings = Embeddings(path="sentence-transformers/nli-mpnet-base-v2", content=True, objects=True)

    # Create an index for the list of text
    embeddings.index(data)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)