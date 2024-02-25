import os
import requests
import base64
import json


def is_app_engine():
    return os.getenv("GAE_ENV", "").startswith("standard")


def get_directory(base_directory):
    if is_app_engine():
        return os.path.join("/tmp", base_directory)
    else:
        return base_directory


def save_video_file(video_file, board_token, directory="uploads"):
    directory = get_directory(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    video_file_path = os.path.join(directory, f"video-{board_token}.mp4")
    with open(video_file_path, "wb") as f:
        for chunk in video_file.chunks():
            f.write(chunk)
    return video_file_path


import os
import wave


def save_audio_file(uploaded_audio, board_token, directory="audio"):
    directory = get_directory(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    audio_file_path = os.path.join(directory, f"audio-{board_token}.wav")
    with open(audio_file_path, "wb") as wf:
        for chunk in uploaded_audio.chunks():
            wf.write(chunk)

    return audio_file_path


import cv2
from imutils import paths


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def find_least_blurry_frame(directory):
    highest_focus_measure = 0
    least_blurry_image_path = None

    for image_path in paths.list_images(directory):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        if fm > highest_focus_measure:
            highest_focus_measure = fm
            print(least_blurry_image_path)
            least_blurry_image_path = image_path
    if least_blurry_image_path is None:
        raise Exception("No frames found")
    return least_blurry_image_path


def extract_and_find_least_blurry_frame(video_file_path, directory="frames"):
    directory = get_directory(directory)
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")
    if not os.path.exists(directory):
        os.makedirs(directory)

    highest_focus_measure = 0
    least_blurry_frame_path = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        if fm > highest_focus_measure:
            highest_focus_measure = fm
            least_blurry_frame_path = os.path.join(
                directory, f"frame-{frame_count}.jpg"
            )
            cv2.imwrite(least_blurry_frame_path, frame)
        frame_count += 1

    cap.release()

    if least_blurry_frame_path is None:
        raise Exception("No frames found")
    for file in os.listdir(directory):
        if file != os.path.basename(least_blurry_frame_path):
            os.remove(os.path.join(directory, file))

    return least_blurry_frame_path


def convert_speech_to_text(audio_file_path, model="whisper-1"):
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    }
    files = {
        "file": ("openai.mp3", open(audio_file_path, "rb"), "audio/mpeg"),
        "model": (None, model),
    }
    response = requests.post(
        "https://api.openai.com/v1/audio/transcriptions", headers=headers, files=files
    )
    if response.status_code == 200:
        response_data = response.json()
        transcript = response_data["text"]
        return transcript
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")


def image_to_text(
    image_path, prompt, current_context, model="gpt-4-vision-preview", max_tokens=50
):
    print(current_context)
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    current_query = {"prompt": prompt, "base64_image": base64_image}
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": convert_conversations_to_payload(current_context, current_query),
        "max_tokens": max_tokens,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    if response.status_code == 200:
        response_data = response.json()
        text_response = response_data["choices"][0]["message"]["content"]
        return text_response
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")


def convert_text_to_speech(
    input_text, board_token, model="tts-1", voice="alloy", directory="response"
):
    if not os.path.exists(directory):
        os.makedirs(directory)
    output_file = os.path.join(directory, f"response-{board_token}.mp3")
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "input": input_text, "voice": voice}
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers=headers,
        json=payload,
        stream=True,
    )
    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=1024):
            yield chunk
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")


import time


def get_time(starting_time):
    return time.time() - starting_time


from google.cloud import vision
import io


def ocr(image_path):
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return texts[0].description.strip()
    else:
        return ""


from uuid import uuid4
from .firebase_config import db, storage, bucket, firestore


def upload_image_to_storage(file_path, board_id):
    original_filename = os.path.basename(file_path)
    extension = os.path.splitext(original_filename)[1]
    filename = f"{uuid4()}{extension}"
    blob = bucket.blob(f"boards/{board_id}/{filename}")
    blob.upload_from_filename(file_path, content_type="image/jpeg")
    blob.make_public()
    return blob.public_url


def board_exists(board_id):
    board_ref = db.collection("boards").document(board_id)
    board_doc = board_ref.get()
    return board_doc.exists


def create_board(board_id):
    board_data = {
        "created_at": firestore.SERVER_TIMESTAMP,
    }
    board_ref = db.collection("boards").document(board_id)
    board_ref.set(board_data)


def add_query_to_board(board_id, query_data):
    if not board_exists(board_id):
        create_board(board_id)
    query_id = generate_query_id()

    store_response(query_id, query_data["response"])
    query_ref = (
        db.collection("boards")
        .document(board_id)
        .collection("queries")
        .document(query_id)
    )
    query_ref.set(query_data)


import random
import string


def generate_query_id():
    timestamp = int(time.time())
    random_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{timestamp}_{random_str}"


def get_recent_queries(board_id, number_of_queries=3):
    query_ref = db.collection("boards").document(board_id).collection("queries")
    query_docs = (
        query_ref.order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(number_of_queries)
        .stream()
    )
    queries = []
    for query in query_docs:
        query_data = query.to_dict()
        query_data["id"] = query.id
        queries.append(query_data)
    return queries


def convert_conversations_to_payload(conversations, current_query):
    payload_messages = []
    payload_messages.append(
        {
            "role": "system",
            "content": "You are yaR, an AI wearable pendent that helps visually impaired people understand their environment. The user will ask you questions about the environment and you will respond with the most relevant information. Your responses should be crisp and to the point. You may reply with questions should you need more context.",
        }
    )
    for conversation in conversations:
        payload_messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": conversation["prompt"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": conversation["image_url"],
                            "detail": "high",
                        },
                    },
                ],
            }
        )
        payload_messages.append(
            {"role": "assistant", "content": conversation["response"]}
        )
    if current_query:
        payload_messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": current_query["prompt"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{current_query['base64_image']}",
                            "detail": "high",
                        },
                    },
                ],
            }
        )

    return payload_messages


from .pinecone_config import index


from transformers import CLIPProcessor, CLIPModel

model_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

import torch


def store_response(response_id, response_text):
    inputs = processor(
        text=[response_text], return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs).numpy()
    index.upsert(vectors=[(response_id, embeddings[0])])


def query_responses(query_text, top_k=5):
    inputs = processor(
        text=[query_text], return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs).numpy()
    query_vector = query_embedding[0]
    response = query_pinecone(query_vector, top_k=top_k)
    print(response[0])
    return response


def query_pinecone(vector, top_k=5, filter=None):
    vector_list = vector.tolist()
    query_response = index.query(vector=vector_list, top_k=top_k, filter=filter)
    return query_response
