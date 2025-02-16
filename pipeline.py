import os
import time
import openai
import cv2
import chromadb
import pandas as pd
from chromadb.config import Settings
from tqdm import tqdm
from openai import APIError, RateLimitError  

os.environ["TOKENIZERS_PARALLELISM"] = "false"

OPENAI_API_KEY = "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"

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(
    path="/Users/jjburrell/TreeHacks/Tesla/chromadb",
    settings=Settings(persist_directory="/Users/jjburrell/TreeHacks/Tesla/chromadb")
)
collection = chroma_client.get_or_create_collection("tesla-videos")

VIDEO_PATH = "/Users/jjburrell/Downloads/tesla-real-world-video-q-a/videos"
QUESTIONS_CSV = "/Users/jjburrell/Downloads/tesla-real-world-video-q-a/questions.csv"
OUTPUT_CSV = "/Users/jjburrell/Downloads/tesla-real-world-video-q-a/answers.csv"
SUBMISSION_SAMPLE = "/Users/jjburrell/Downloads/tesla-real-world-video-q-a/submission_sample.csv"


def handle_rate_limit(error, retries=5):
    wait_time = 10  # Start with 10 seconds
    for i in range(retries):
        print(f"⏳ Rate limit hit! Retrying in {wait_time} seconds...")
        time.sleep(wait_time)
        wait_time *= 2  


def answer_questions():
    """Answer questions using GPT-4o while handling rate limits."""
    print("🤖 Answering Questions with GPT-4o...")

    df = pd.read_csv(QUESTIONS_CSV)
    answers = []

    for index, row in df.iterrows():
        try:
            question = row["question"]

            results = collection.query(
                query_texts=[question],
                n_results=3
            )

            video_context = " ".join([doc for doc in results["documents"][0]])

            while True:
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": f"Question: {question}\nContext: {video_context}"}]
                    )
                    break  
                except RateLimitError as e:  
                    handle_rate_limit(e)
                except APIError as e:
                    print(f"❌ OpenAI API Error: {e}")
                    answers.append({"question_id": index + 1, "answer": "API Error"})
                    break  # Skip to next question if an API error occurs

            answer = response.choices[0].message.content
            answers.append({"question_id": index + 1, "answer": answer})

            if (index + 1) % 3 == 0:
                print("⏳ Waiting 20s to respect OpenAI rate limits...")
                time.sleep(20)

        except Exception as e:
            print(f"❌ Unexpected Error answering question {index}: {e}")
            answers.append({"question_id": index + 1, "answer": "Error"})

    submission_df = pd.DataFrame(answers)

    sample_df = pd.read_csv(SUBMISSION_SAMPLE)

    if set(sample_df.columns) == set(submission_df.columns):
        submission_df = submission_df[sample_df.columns]

    submission_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Answers saved to {OUTPUT_CSV} in submission format!")

if __name__ == "__main__":
    answer_questions()
