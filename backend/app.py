import os
import chromadb

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document


# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"  # Replace with your OpenAI API key

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# File path to CSV
file_path = "./healthcare_dataset.csv" # Replace with your actual file path

# Load CSV
df = pd.read_csv(file_path)

# Combine all row data into one text field per row
def row_to_text(row):
    return " | ".join(row.astype(str))  # Convert row values to strings and join them

# Convert each row into a single text entry
texts = df.apply(row_to_text, axis=1).tolist()

# Convert to LangChain Documents
documents = [Document(page_content=text) for text in texts]

# Split text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Initialize Chroma remote client
client = chromadb.HttpClient(
    ssl=True,
    host='api.trychroma.com',
    tenant='eb7e581d-5a59-4a30-b990-a6cb9a873874',  # Replace with your tenant ID
    database='health-history',
    headers={'x-chroma-token': 'ck-BjgunzcHw8gg8K7YqBQ2HwWP3ARAWHg4bYEjG5fhSYGm'}  # Replace with your Chroma token
)

# Initialize OpenAI embeddings
embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")

# Store documents in Chroma (Batch processing)
vectorstore = Chroma(
    client=client,
    collection_name="healthcare_data",
    embedding_function=embedding_function
)

# ChromaDB limit: process documents in batches of 100
batch_size = 100
for i in range(0, len(docs), batch_size):
    batch = docs[i:i + batch_size]
    vectorstore.add_documents(batch)
    print(f"Inserted batch {i // batch_size + 1} of {len(docs) // batch_size + 1}")

# Initialize Chat Model
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

# Define Retrieval Chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# Define API route to handle queries
@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get('query', '')

    if query_text:
        response = qa_chain.run(query_text)
        return jsonify({'response': response})
    else:
        return jsonify({'response': 'No query provided'})

# Start Flask app
if __name__ == '__main__':
    app.run(debug=True)
