import cv2
import insightface
import numpy as np
import pickle
from numpy.linalg import norm

app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# Load regstered face
with open("dev_face.pkl", "rb") as f:
    known = pickle.load(f)

def similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

cam = cv2.VideoCapture(0)
print("Searching for face...")

while True:
    ret, frame = cam.read()
    faces = app.get(frame)

    for face in faces:
        score = similarity(known, face.embedding)

        if score > 0.5:  
            label = f"Welcome Dev! ({score:.2f})"
            color = (0, 255, 0)
        else:
            label = f"Not recognized ({score:.2f})"
            color = (0, 0, 255)

        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]),
                     (box[2], box[3]), color, 2)
        cv2.putText(frame, label, (box[0], box[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 4)

    cv2.imshow("IndID Login", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()