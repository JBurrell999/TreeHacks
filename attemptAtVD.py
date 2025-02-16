import os
import csv
import cv2
import torch
import clip
import pandas as pd
import numpy as np
from PIL import Image
import chromadb

# Initialize ChromaDB client
client = chromadb.HttpClient(
    ssl=True,
    host='api.trychroma.com',
    tenant='a9804358-3bb6-4476-a5dc-17f85c919323',
    database='TreeHacks',
    headers={'x-chroma-token': 'ck-5AEeNF5BPm4Ci1ocb4uVZAmnvHkEhVjE4Yp3raETDFKQ'}
)

# Create or get collection
collection = client.get_or_create_collection(name="video_question_embeddings")

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Paths
video_dir = "/Users/jjburrell/Downloads/tesla-real-world-video-q-a/videos/videos"
questions_path = "/Users/jjburrell/Downloads/tesla-real-world-video-q-a/questions.csv"

# Function to extract embeddings from video frames
def extract_video_embedding(video_path, model, preprocess, device):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(0, frame_count, max(1, frame_count // 5)):  # Sample 5 frames per video
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            frames.append(preprocess(image).unsqueeze(0).to(device))
    cap.release()
    if frames:
        frames_tensor = torch.cat(frames)
        with torch.no_grad():
            embeddings = model.encode_image(frames_tensor)
        return embeddings.mean(dim=0).cpu().numpy()
    return None

# Function to extract embeddings from text
def extract_text_embedding(text, model, device):
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
    return text_embedding.cpu().numpy()

# Process videos and upload embeddings to ChromaDB
for video_file in os.listdir(video_dir):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(video_dir, video_file)
        embedding = extract_video_embedding(video_path, model, preprocess, device)
        if embedding is not None:
            collection.add(
                ids=[video_file],  # Use filename as ID
                embeddings=[embedding.tolist()],
                metadatas=[{"filename": video_file}]
            )
            print(f"Uploaded: {video_file}")

# Process questions and generate submission
questions_df = pd.read_csv(questions_path)
submission = []

for index, row in questions_df.iterrows():
    question_id = row['question_id']
    question_text = row['question']
    options = [row['option_1'], row['option_2'], row['option_3'], row['option_4']]

    # Extract question embedding
    question_embedding = extract_text_embedding(question_text, model, device)

    # Query ChromaDB to find the most similar video
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=1
    )

    # Assuming the most similar video is the correct answer
    if results and results['ids']:
        best_match_video = results['ids'][0][0]  # Get the first result's ID
        # Map the best match video to the correct option
        for i, option in enumerate(options, start=1):
            if option in best_match_video:
                submission.append([question_id, i])  # Append question_id and option number
                break

# Save submission to CSV
submission_df = pd.DataFrame(submission, columns=['question_id', 'answer'])
submission_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
