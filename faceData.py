#saving face images+names 
import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []
dataset_path = "./data/"
file_name = input("Enter the name of the Person: ")

while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    
    if len(faces) == 0:
        continue
        
    # Pick the largest face in the frame (closest to camera)
    face = sorted(faces, key=lambda f: f[2]*f[3])[-1]
    
    x, y, w, h = face
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Crop and Save the Face Selection
    offset = 10
    face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
    face_section = cv2.resize(face_section, (100, 100))

    # Store every 10th frame to add variety
    if len(face_data) % 10 == 0:
        face_data.append(face_section)
        print(f"Captured {len(face_data)}/20") # Feedback for user

    cv2.imshow("Face Section", face_section)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or len(face_data) >= 20: # Stop after 20 faces
        break

# Convert to numpy array
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# Ensure data directory exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Save the data
np.save(dataset_path + file_name + '.npy', face_data)
print("Data Successfully saved at " + dataset_path + file_name + '.npy')

cap.release()
cv2.destroyAllWindows()