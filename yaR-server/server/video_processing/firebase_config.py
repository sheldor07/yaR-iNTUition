import firebase_admin
from firebase_admin import credentials, firestore, storage

cred = credentials.Certificate("video_processing/yar-v2-firebase-adminsdk-fbkkl-d3d32b85e1.json")

firebase_admin.initialize_app(cred, {"storageBucket": "yar-v2.appspot.com"})

db = firestore.client()
bucket = storage.bucket()
