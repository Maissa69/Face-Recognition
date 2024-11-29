import sqlite3
import cv2
import os
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Constants
#MESSAGE = "WELCOME  Instruction: to register your attendance kindly click on 'a' on keyboard"
DATETODAY = date.today().strftime("%m_%d_%y")
ATTENDANCE_FILE = f'Attendance/Attendance-{DATETODAY}.csv'

# Initialize directories and attendance file
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

if not os.path.isfile(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w') as f:
        f.write('Name,Roll,Time')

# Initialize face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to access webcam
def get_video_capture():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open webcam.")
        return cap
    except Exception as e:
        print(f"Error accessing webcam: {e}")
        return None

# Get total registered users
def total_registered_users():
    return len(os.listdir('static/faces'))

# Extract faces from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# Train the face recognition model
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(np.array(faces), labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Extract today's attendance data
def extract_attendance():
    df = pd.read_csv(ATTENDANCE_FILE)
    return df['Name'], df['Roll'], df['Time'], len(df)

# Add attendance entry for a specific user
def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(ATTENDANCE_FILE)
    
    if str(userid) not in list(df['Roll']):
        with open(ATTENDANCE_FILE, 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
        print(f"Attendance marked for {username}, at {current_time}")
    else:
        print("This user has already marked attendance for the day.")

# Main attendance marking function
def mark_attendance():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        print('This face is not registered with us, kindly register yourself first.')
        return 'Face not in database.'

    cap = get_video_capture()
    
    if cap is None:
        return "Failed to access webcam."

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        faces = extract_faces(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            cv2.putText(frame, f'{identified_person}', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)

            if cv2.waitKey(1) == ord('a'):
                add_attendance(identified_person)
                break

        cv2.imshow('Attendance Check - Press "q" to exit', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print("Attendance registered successfully.")
    return 'Attendance taken successfully.'

# Function to add a new user with error handling
def add_new_user():
    newusername = input("Enter username: ")
    newuserid = input("Enter user ID: ")
    
    if not newusername or not newuserid.isnumeric():
        print("Invalid username or user ID.")
        return
    
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    
    os.makedirs(userimagefolder, exist_ok=True)

    cap = get_video_capture()
    
    if cap is None:
        return

    i, j = 0, 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        faces = extract_faces(frame)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0 ,20), 2)

            if j % 10 == 0 and i < 50: # Capture up to 50 images
                name = f'{newusername}_{i}.jpg'
                cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y + h,x:x + w])
                i += 1
            
            j += 1
        
        if i >= 50 or j >= 500: # Stop after capturing enough images or after many attempts
            break
        
        cv2.imshow('Adding New User', frame)
        
        if cv2.waitKey(1) == 27: # Escape key to exit early
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print('Training Model...')
    train_model()

    names, rolls, times, l = extract_attendance()
    
    print("User added successfully.")
    
    return names.tolist(), rolls.tolist(), times.tolist(), l

# Main execution loop
def main():
    while True:
        print("\n1. Mark Attendance")
        print("2. Add New User")
        print("3. Exit")
        
        choice = input("Enter your choice: ")

        if choice == '1':
            message = mark_attendance()
            print(message)
        elif choice == '2':
            names, rolls, times, l = add_new_user()
            print("Updated User List:")
            print(f"Names: {names}")
            print(f"Rolls: {rolls}")
            print(f"Times: {times}")
            print(f"Total Users: {l}")
        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
