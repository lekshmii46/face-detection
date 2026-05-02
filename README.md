Attendance System using Face Recognition

A real-time AI-powered attendance system built with Streamlit, OpenCV, and Machine Learning that automatically detects faces and records attendance without manual input.

🌟 Overview

This project replaces traditional attendance methods with a contactless, automated system. It captures student faces via webcam, trains a machine learning model, and marks attendance instantly when a face is recognized.

✨ Key Highlights
🎥 Real-time face detection using OpenCV
🧑‍💻 Easy student registration via webcam
🧠 Machine Learning model (KNN) for recognition
📅 Daily attendance logs saved automatically
📥 Downloadable attendance reports (CSV)
⚡ Fast and lightweight Streamlit interface
🧰 Tech Stack
Category	Tools Used
Frontend UI	Streamlit
Computer Vision	OpenCV
ML Algorithm	KNN (Scikit-learn)
Data Handling	Pandas, NumPy
Model Storage	Joblib
📁 Directory Structure
.
├── app.py
├── dataset/
│   └── student_name_id/
│       └── images...
├── attendance_records/
│   └── Attendance_YYYY-MM-DD.csv
├── model.pkl
⚙️ Setup Instructions
1️⃣ Clone the Project
git clone <repository-url>
cd face-attendance-system
2️⃣ Install Required Libraries
pip install streamlit opencv-python numpy pandas scikit-learn joblib
▶️ Run the Application
streamlit run app.py
📌 Usage Guide
🔐 Register a Student
Input Name and Student ID
System captures 100 face samples
Automatically trains the recognition model
🎯 Start Live Attendance
Enable webcam
System detects & recognizes faces
Marks attendance instantly
Unknown faces are ignored
📊 View Attendance Reports
Select date-wise records
View structured table
Export as CSV file
🧠 Behind the Scenes
Faces are detected using Haar Cascade Classifier
Each face is:
Converted to grayscale
Resized to 50×50 pixels
Flattened into feature vectors
KNN algorithm is trained on these vectors
During recognition:
Nearest neighbors are checked
Distance threshold determines if face is known or unknown
📄 Sample Attendance Record
Name,Time,Date
John_101,09:12:34,2026-05-02
⚠️ Important Notes
Good lighting improves accuracy
Avoid multiple faces during registration
Ensure webcam permissions are enabled
Model retrains after each new registration
