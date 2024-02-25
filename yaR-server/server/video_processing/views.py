from django.shortcuts import render
import time

# Create your views here.
from django.http import JsonResponse, FileResponse, HttpResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import os
from concurrent.futures import ThreadPoolExecutor
from .firebase_config import firestore


from .utils import (
    save_audio_file,
    save_video_file,
    get_recent_queries,
    find_least_blurry_frame,
    convert_speech_to_text,
    upload_image_to_storage,
    image_to_text,
    convert_text_to_speech,
    add_query_to_board,
    get_time,
    query_responses,
    extract_and_find_least_blurry_frame,
    ocr,
)


@csrf_exempt
def hello_world(request):
    return HttpResponse("<h1>Hello, World!</h1>")


@csrf_exempt
def ocr_view(request):
    if request.method == "POST":
        board_token = request.headers.get("X-Token")
        if not board_token:
            return HttpResponse({"message": "Token is required"}, status=400)

        video_file = request.FILES.get("video")
        audio_file = request.FILES.get("audio")
        if not video_file:
            return HttpResponse({"message": "Video is required"}, status=400)
        if not audio_file:
            return HttpResponse({"message": "Audio is required"}, status=400)

        start_time = time.time()
        video_file_path = save_video_file(video_file, board_token)
        print("Video file saved", get_time(start_time))
        audio_file_path = save_audio_file(audio_file, board_token)
        previous_op_time = time.time()
        current_context = get_recent_queries(board_token)
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_frame = executor.submit(
                extract_and_find_least_blurry_frame, video_file_path
            )
            future_transcript = executor.submit(convert_speech_to_text, audio_file_path)

            least_blurry_frame = future_frame.result()
            print("Least blurry frame found", get_time(previous_op_time))
            previous_op_time = time.time()

            transcript = future_transcript.result()

            print("Transcript made ", get_time(previous_op_time))

            previous_op_time = time.time()
        current_context += query_responses(transcript)
        extracted_text = ocr(least_blurry_frame)
        print("Extracted text", get_time(previous_op_time))
        vision_prompt = f"Image Description: {extracted_text}\nUser Prompt: {transcript}\n\nPlease provide a detailed response based on the image description and the user's prompt."
        previous_op_time = time.time()
        vision_response = image_to_text(
            least_blurry_frame, extracted_text, transcript, current_context
        )
        print("Got Vision response", get_time(previous_op_time))
        previous_op_time = time.time()
        audio_response_file_path = convert_text_to_speech(vision_response, board_token)
        print("Text to speech completed", get_time(previous_op_time))
        print("Total time taken: ", get_time(start_time))
        return FileResponse(open(audio_response_file_path, "rb"))
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)


@csrf_exempt
def upload_video(request):
    if request.method == "POST":
        board_token = request.headers.get("X-Token")
        if not board_token:
            return HttpResponse({"message": "Token is required"}, status=400)

        video_file = request.FILES.get("video")
        print("Received video file", video_file)
        audio_file = request.FILES.get("audio")
        start_time = time.time()
        print("Received video and audio files - Timer started")

        if not video_file:
            return HttpResponse({"message": "Video is required"}, status=400)
        if not audio_file:
            return HttpResponse({"message": "Audio is required"}, status=400)

        try:
            video_file_path = save_video_file(video_file, board_token)
            print("Video file saved, Time taken: ", get_time(start_time))
            previous_op_time = time.time()

            audio_file_path = save_audio_file(audio_file, board_token)
            print("Audio file saved, Time taken: ", get_time(previous_op_time))
            previous_op_time = time.time()
            current_context = get_recent_queries(board_token)
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_frame = executor.submit(
                    extract_and_find_least_blurry_frame, video_file_path
                )
                future_transcript = executor.submit(
                    convert_speech_to_text, audio_file_path
                )

                least_blurry_frame = future_frame.result()
                print("Least blurry frame found", get_time(previous_op_time))
                previous_op_time = time.time()
                transcript = future_transcript.result()
                print("Transcript made", get_time(previous_op_time))
                previous_op_time = time.time()

                future_vision_response = executor.submit(
                    image_to_text, least_blurry_frame, transcript, current_context
                )

                future_image_upload = executor.submit(
                    upload_image_to_storage,
                    least_blurry_frame,
                    board_token,
                )

                vision_response = future_vision_response.result()
                print("Vision response generated", get_time(previous_op_time))
                previous_op_time = time.time()

                image_url = future_image_upload.result()
                print("Image uploaded", get_time(previous_op_time))
                previous_op_time = time.time()
            audio_stream = convert_text_to_speech(vision_response, board_token)
            print("Text to speech completed, Time taken: ", get_time(previous_op_time))
            add_query_to_board(
                board_token,
                {
                    "prompt": transcript,
                    "response": vision_response,
                    "created_at": firestore.SERVER_TIMESTAMP,
                    "image_url": image_url,
                },
            )
            print("Query saved", get_time(previous_op_time))
            print("Total time taken: ", get_time(start_time))
            response = StreamingHttpResponse(audio_stream, content_type="audio/mpeg")
            os.remove(video_file_path)
            os.remove(audio_file_path)
            os.remove(least_blurry_frame)
            return response
        except Exception as e:
            print(f"Error: {e}")
            return HttpResponse({"message": "An error occurred"}, status=500)

        except Exception as e:
            return JsonResponse({"message": str(e)}, status=500)
