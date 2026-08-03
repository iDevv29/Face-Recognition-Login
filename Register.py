import cv2
import insightface
import numpy as np

from database import init_db, save_user
init_db()

app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

name = input("Enter your name: ").strip()

cam = cv2.VideoCapture(0)
print(f"Registering {name}... Look at the camera.")

encodings = []
count = 0

while count < 30:
    ret, frame = cam.read()
    faces = app.get(frame)
    
    if faces:
        encoding = faces[0].embedding
        encodings.append(encoding)
        count += 1
        
        face = faces[0]
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), 
                     (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"Capturing... {count}/30", 
                   (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
    
    cv2.imshow("IndID - Registration", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

if encodings:
    avg_encoding = np.mean(encodings, axis=0)
    save_user(name, avg_encoding)