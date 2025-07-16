import streamlit as st
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from time import sleep

# Load your model and classifier
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./model.keras')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion(frame):
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]

        if np.sum([roi_gray])!=0:
            roi = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_AREA)
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            
            roi = roi.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
            
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            return frame, label
        else:
            return frame, 'No Faces'

    return frame, 'No Faces'

st.title("Emotion Detection")

# Create a placeholder for the webcam feed
video_placeholder = st.empty()

# Create a placeholder for the mood text
mood_placeholder = st.empty()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame from webcam.")
        break

    frame, mood = detect_emotion(frame)
    
    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the webcam feed
    video_placeholder.image(frame_rgb, channels="RGB")
    
    # Update the mood text
    mood_placeholder.text(f"Detected Mood: {mood}")
    
    sleep(0.1)  # Add a small delay to reduce CPU usage

cap.release()