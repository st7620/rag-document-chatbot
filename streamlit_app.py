import streamlit as st
from transformers import pipeline

# App title
st.set_page_config(page_title="Document Chatbot")

# File uploader widget
uploaded_files = st.file_uploader("Choose an image file", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.write(bytes_data)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating response
def generate_response(file_name, prompt_input):                       
    nlp = pipeline("document-question-answering",
                   model="impira/layoutlm-document-qa",
                   )
    return nlp(file_name, prompt_input)['answer']

#Temp file name
file_name = "https://templates.invoicehome.com/invoice-template-us-neat-750px.png"

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(file_name, prompt) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)