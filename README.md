# Document Chatbot

This is an end-to-end self-hosted chatbot application that answers questions about user-uploaded documents, created with Break Through Tech AI @ Cornell Tech and JPMorgan Chase. This chatbot utilizes the RAG framework to store PDF documents using an embedding model and a vector database. Using semantic search, it then selects the document that is most relevant to the user's query and provides this document as enhanced context with the query for an pre-trained LLM to generate an answer.

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-document-chatbot-3m33nueb3aqbexlyfaucra.streamlit.app/)