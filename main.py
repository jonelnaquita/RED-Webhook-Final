# main.py

import os
import pinecone
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from flask import Flask, request, jsonify
from pymongo import MongoClient
from datetime import datetime

load_dotenv()

# Load environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MONGODB_URL = os.getenv("MONGODB_URL")

# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Initialize Flask app
app = Flask(__name__)

# MongoDB setup
client = MongoClient(MONGODB_URL)
db = client.get_database('Dialogflow+MongoDB')
collection = db.get_collection('conversations')

conversation_history = []

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

"""
def esReceiveMessage():
    try:
        data = request.get_json()
        query_text = data['queryResult']['queryText']

        # Append user query to the conversation history
        conversation_history.append(f'User: {query_text}')

        # Combine conversation history for OpenAI input
        conversation_input = '\n'.join(conversation_history)

        result = retrieval_answer(conversation_input, doc_db, llm)
        fulfillmentText = result['response']

        if result['status'] == 1:
            # Append AI response to the conversation history
            conversation_history.append(f'AI: {result["response"]}')

            return jsonify({
                'fulfillmentText': result['response']
            })
    except:
        pass
    return jsonify({
        'fulfillmentText': 'Something went wrong.'
    })
"""

@app.route('/dialogflow', methods=['POST'])
def handle_dialogflow():
    data = request.get_json()
    
    # Extract relevant data from the Dialogflow webhook response
    responseID = data['responseId']
    query_text = data['queryResult']['queryText']
    fulfillment_text = data['queryResult'].get('fulfillmentText', '')
    action = data['queryResult']['action']
    intent_name = data['queryResult']['intent']['displayName']
    timestamp = datetime.now()

    # Create a document to save in MongoDB
    conversation = {
        'responseId': responseID,
        'queryText': query_text,
        'fulfillmentText': fulfillment_text,
        'action': action,
        'intentName': intent_name,
        'timestamp': timestamp
    }

    # Insert the conversation document into the collection
    collection.insert_one(conversation)
    
    try:
        action = data['queryResult']['action']
        if action == 'input.unknown':
            # Call retrieval_answer to get the response
            query = data['queryResult']['queryText']
            response = retrieval_answer(query, doc_db, llm)
            
            # Return the response to Dialogflow
            return jsonify({
                'fulfillmentText': response
            })
        else:
            # Handle other actions here if needed
            return jsonify({
                'fulfillmentText': 'Action not recognized.'
            })
    except:
        return jsonify({
            'fulfillmentText': 'Error occurred.'
        })

if __name__ == "__main__":
    # Initialize Langchain and start the Flask app
    llm = ChatOpenAI()
    doc_db = embedding_db()
    app.run()