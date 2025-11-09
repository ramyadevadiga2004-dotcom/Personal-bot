# Personal-bot
Mini Jarvis — AI Assistant with LLaMA, Pinecone & Gradio

Mini Jarvis is a locally running AI-powered assistant that allows you to ask intelligent questions from your own documents, such as PDFs, using advanced natural language understanding. It integrates multiple technologies to make a smart, private, and efficient AI system. This project combines a local LLaMA/Mistral model (using llama.cpp) for text generation, Pinecone as a vector database for semantic search, and Sentence Transformers to convert text into embeddings for retrieval. The interface is built with Gradio, making it simple and interactive to chat with your data directly from your browser.

The project is designed to work completely offline once initialized, ensuring your data stays private. Users can upload their study materials, project notes, or PDFs, and Mini Jarvis will analyze them, store the text as embeddings in Pinecone, and later use the LLaMA model to generate context-aware answers based on that stored information.

To set up, clone the repository, create a virtual environment, and install dependencies. You’ll need to set your Pinecone API key and the path to your GGUF model in a .env file. Then, initialize your Pinecone index, ingest your PDF data, and launch the chat app using Gradio. The chat interface allows you to enter any question, retrieves relevant document chunks using Pinecone, and generates answers using the local LLaMA model.

The folder structure includes scripts for different purposes: pinecone_setup.py initializes the Pinecone index, ingest.py processes and uploads PDF content, app_gradio.py runs the Gradio chat interface, and test_llama.py ensures your model loads correctly. All environment settings are stored in a .env file for easy configuration.

This project helps you understand how retrieval-augmented generation (RAG) works — combining semantic search and language generation. It can be extended to support voice-based interaction, multi-document search, or integration with cloud-based AI services.

Mini Jarvis is developed by Ramya Manjunatha, an MCA final-year student passionate about artificial intelligence and machine learning. The project is licensed under the MIT License, so you’re free to use, modify, and share it.
