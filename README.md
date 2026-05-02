#Face Recognition Attendance System

This is a simple web application that uses face recognition to mark attendance automatically using a webcam.

Features
Register students using webcam
Capture face images
Train model automatically
Mark attendance in real time
Save attendance in CSV file
View and download reports
Technologies Used
Python
Streamlit
OpenCV
Scikit-learn (KNN)
Pandas, NumPy
How to Run
Install required libraries:
pip install streamlit opencv-python numpy pandas scikit-learn joblib
Run the app:
streamlit run app.py
How to Use
Go to Register Student
Enter name and ID
Capture face images
Model will train automatically

Then:

Go to Live Attendance
Start webcam
Attendance will be marked automatically
Output
Attendance is saved in:
attendance_records/
Format:
Name | Time | Date
Notes
Make sure webcam is working
Good lighting improves accuracy
Register each student clearly
