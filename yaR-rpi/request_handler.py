import requests
from pathlib import Path


def upload_video_and_handle_response(video_path, audio_path, token):
    url = "http://47.128.198.46:8000/video_processing/upload/"
    print("video_path", video_path)
    print("audio_path", audio_path)
    print("token", token)

    headers = {"X-Token": token}
    video_file_name = Path(video_path).name
    audio_file_name = Path(audio_path).name

    if not Path(video_path).is_file():
        print(f"Error: The video file {video_path} does not exist.")
        return
    if not Path(audio_path).is_file():
        print(f"Error: The audio file {audio_path} does not exist.")
        return

    with open(video_path, "rb") as video_file, open(audio_path, "rb") as audio_file:
        files = {
            "video": (video_file_name, video_file, "video/mp4"),
            "audio": (audio_file_name, audio_file, "audio/wav")
        }
        response = requests.post(url, headers=headers, files=files, stream=True)

    if response.status_code == 200:
        mp3_path = video_path.rsplit(".", 1)[0] + ".mp3"
        with open(mp3_path, "wb") as mp3_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  
                    mp3_file.write(chunk)
        print(f"MP3 saved to {mp3_path}")
        return mp3_path
    else:
        print(f"Error: {response.status_code} - {response.text}")

