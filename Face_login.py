import cv2
import insightface
import numpy as np
from numpy.linalg import norm
from database import get_all_users

app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

users = get_all_users()
print(f"{len(users)} user(s) loaded.")

THRESHOLD = 0.5

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

cam = cv2.VideoCapture(0)
print("Searching for face...")

while True:
    ret, frame = cam.read()
    faces = app.get(frame)

    for face in faces:
        embedding = face.embedding
        bbox = face.bbox.astype(int)

        best_score = 0
        best_match = "Unknown"

        for name, known_enc in users:
            score = cosine_similarity(embedding, known_enc)
            if score > best_score:
                best_score = score
                best_match = name

        if best_score > THRESHOLD:
            label = f"Welcome {best_match}! ({best_score:.2f})"
            color = (0, 255, 0)
        else:
            label = f"Not recognized ({best_score:.2f})"
            color = (0, 0, 255)

        cv2.rectangle(frame, (bbox[0], bbox[1]),
                     (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 4)

    cv2.imshow("IndID Login", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()