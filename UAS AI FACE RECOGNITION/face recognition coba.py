import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime

# Mendapatkan direktori tempat skrip berada
script_dir = os.path.dirname(os.path.abspath(__file__))

# Menentukan path gambar
path = os.path.join(script_dir, "Imageattendance")

images = []
classNames = []
mylist = os.listdir(path)

for cl in mylist:
    if cl.endswith(('.jpg', '.jpeg', '.png')):
        curImg = cv2.imread(os.path.join(path, cl))
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

print(f"{cl} in mylist:")

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

# Pastikan untuk mengubah path untuk file CSV
attendance_file_path = os.path.join(script_dir, 'Attendance.csv')

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open(attendance_file_path, 'a+') as f:
        # Pindahkan kursor ke awal file agar bisa dibaca
        f.seek(0)
        myDataList = f.readlines()
        nameList = [entry.split('|')[0].strip() for entry in myDataList]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'\n{name} | {dtString}')

# Mendapatkan encoding wajah dari gambar
encodeListKnow = findEncodings(images)
print(len(encodeListKnow), 'Encoding Complete')

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
