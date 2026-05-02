import streamlit as st
import cv2
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Constants
DATASET_DIR = "dataset"
ATTENDANCE_DIR = "attendance_records"
MODEL_PATH = "model.pkl"
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

st.set_page_config(page_title="Face Detection Attendance", layout="wide")

st.title("Face Detection-Based Attendance System")

# Load face detector
@st.cache_resource
def get_face_detector():
    return cv2.CascadeClassifier(HAAR_CASCADE_PATH)

face_detector = get_face_detector()

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    return faces, gray

def train_model():
    faces_data = []
    labels = []
    
    if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
        st.warning("No dataset found. Please register students first.")
        return False
        
    for student_folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, student_folder)
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (50, 50))
                    faces_data.append(img_resized.flatten())
                    labels.append(student_folder)
                    
    if len(faces_data) > 0:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces_data, labels)
        joblib.dump(knn, MODEL_PATH)
        return True
    return False

def mark_attendance(name):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    file_path = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")
    
    # Initialize or load session state for marked attendance
    if 'marked_today' not in st.session_state:
        st.session_state.marked_today = set()
        
    # Check if the file exists and load already marked users if any
    if os.path.isfile(file_path):
        df_existing = pd.read_csv(file_path)
        existing_names = set(df_existing['Name'].tolist())
        st.session_state.marked_today.update(existing_names)
        
    if name not in st.session_state.marked_today:
        df = pd.DataFrame([[name, time_str, date_str]], columns=['Name', 'Time', 'Date'])
        if not os.path.isfile(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)
        st.session_state.marked_today.add(name)
        return True
    return False

menu = ["Live Attendance", "Register Student", "View Reports"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Register Student":
    st.subheader("Register a New Student")
    student_name = st.text_input("Enter Student Name:")
    student_id = st.text_input("Enter Student ID:")
    
    if st.button("Start Registration"):
        if student_name and student_id:
            folder_name = f"{student_name}_{student_id}"
            user_folder = os.path.join(DATASET_DIR, folder_name)
            os.makedirs(user_folder, exist_ok=True)
            
            st.info(f"Starting webcam to capture faces for {student_name}... Please look at the camera.")
            
            cap = cv2.VideoCapture(0)
            count = 0
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            while count < 100:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture from webcam.")
                    break
                    
                faces, gray = extract_faces(frame)
                
                for (x, y, w, h) in faces:
                    count += 1
                    face_img = gray[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (50, 50))
                    cv2.imwrite(os.path.join(user_folder, f"{count}.jpg"), face_img)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Capturing: {count}/100", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                # Mirror the frame horizontally for display
                display_frame = cv2.flip(frame, 1)
                frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                progress_bar.progress(count / 100)
                
            cap.release()
            frame_placeholder.empty()
            st.success(f"Successfully captured 100 images for {student_name}!")
            
            with st.spinner("Training model with new data..."):
                if train_model():
                    st.success("Model trained successfully! You can now mark attendance.")
                else:
                    st.error("Model training failed.")
        else:
            st.warning("Please enter both Name and ID.")

elif choice == "Live Attendance":
    st.subheader("Live Attendance Marking")
    
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Please register students and train the model first.")
    else:
        try:
            knn = joblib.load(MODEL_PATH)
            
            run = st.checkbox("Start WebCam")
            frame_placeholder = st.empty()
            
            if run:
                cap = cv2.VideoCapture(0)
                while cap.isOpened() and run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture from webcam.")
                        break
                        
                    faces, gray = extract_faces(frame)
                    
                    for (x, y, w, h) in faces:
                        face_img = gray[y:y+h, x:x+w]
                        face_img_resized = cv2.resize(face_img, (50, 50)).flatten().reshape(1, -1)
                        
                        # Predict
                        name_id = knn.predict(face_img_resized)[0]
                        
                        # Calculate distance to nearest neighbor to filter out unknowns (optional thresholding)
                        distances, indices = knn.kneighbors(face_img_resized)
                        if distances[0][0] > 4000: # Adjust threshold if needed
                            display_name = "Unknown"
                            color = (0, 0, 255)
                        else:
                            display_name = name_id
                            color = (255, 0, 0)
                            # Mark attendance
                            if mark_attendance(name_id):
                                st.success(f"Attendance marked for {name_id}!")
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, display_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
                    # Mirror the frame horizontally for display
                    display_frame = cv2.flip(frame, 1)
                    frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
                if cap is not None:
                    cap.release()
        except Exception as e:
            st.error(f"Error loading model or running inference: {e}")

elif choice == "View Reports":
    st.subheader("Attendance Reports")
    
    if not os.path.exists(ATTENDANCE_DIR) or len(os.listdir(ATTENDANCE_DIR)) == 0:
        st.info("No attendance records found.")
    else:
        files = sorted(os.listdir(ATTENDANCE_DIR), reverse=True)
        selected_file = st.selectbox("Select Date to View:", files)
        
        if selected_file:
            df = pd.read_csv(os.path.join(ATTENDANCE_DIR, selected_file))
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=selected_file,
                mime='text/csv',
            )
