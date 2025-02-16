import chromadb
import pandas as pd
import clip
import torch
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client
client = chromadb.HttpClient(
    ssl=True,
    host='api.trychroma.com',
    tenant='a9804358-3bb6-4476-a5dc-17f85c919323',
    database='TreeHacks',
    headers={'x-chroma-token': 'ck-5AEeNF5BPm4Ci1ocb4uVZAmnvHkEhVjE4Yp3raETDFKQ'}
)

# Load text embedding model (SentenceTransformer for text queries)
device = "cuda" if torch.cuda.is_available() else "cpu"
text_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# Load questions from CSV
questions_path = "/Users/jjburrell/Downloads/tesla-real-world-video-q-a/questions.csv"
df = pd.read_csv(questions_path)

# Extract question text
questions = df['question'].tolist()

# Generate embeddings for questions
question_embeddings = text_model.encode(questions, convert_to_tensor=True)

# Query ChromaDB and get the best answer
collection = client.get_collection("video_embeddings")
answers = []
for i, question_embedding in enumerate(question_embeddings):
    results = collection.query(
        query_embeddings=[question_embedding.cpu().numpy()],
        n_results=1  # Get the closest match
    )
    
    # Extract the best match and store the answer (assumes metadata contains the correct choice)
    best_match = results['documents'][0][0]
    answers.append(best_match)  # Assuming best_match contains the correct multiple-choice answer

# Save results to a new CSV
output_df = pd.DataFrame({'answer': answers})
output_path = "answers.csv"
output_df.to_csv(output_path, index=False)

print(f"Answers saved to {output_path}")
