# main.py

import os
import pinecone
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def doc_preprocessing():
    loader = DirectoryLoader(
        'data/',
        glob='**/*.pdf',  # Only the PDFs
        show_progress=True
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split

def embedding_db():
    # We use the OpenAI embedding model
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    docs_split = doc_preprocessing()
    doc_db = Pinecone.from_documents(
        docs_split,
        embeddings,
        index_name='red-chatbot'
    )
    return doc_db

def retrieval_answer(query, doc_db, llm):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=doc_db.as_retriever(),
    )
    result = qa.run(query)
    return result

def main():
    st.title("Chat with PDF")
    st.subheader("Developed by Jonel Naquita")

    text_input = st.text_input("Ask your query...")
    if st.button("Ask Query"):
        if len(text_input) > 0:
            st.info("Your Query: " + text_input)
            llm = ChatOpenAI()
            doc_db = embedding_db()
            answer = retrieval_answer(text_input, doc_db, llm)
            st.success(answer)

if __name__ == "__main__":
    main()
