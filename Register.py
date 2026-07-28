import cv2
import insightface
import numpy as np
import pickle

app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

def register_face():
    cam = cv2.VideoCapture(0)
    encodings = []
    frame_count = 0

    print("Slowly move left, right, up, down.")

    while frame_count < 150:
        ret, frame = cam.read()
        faces = app.get(frame)

        if faces:
            encodings.append(faces[0].embedding)
            frame_count += 1
            cv2.putText(frame, f"Capturing... {frame_count}/150",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2)

        cv2.imshow("IndID - Register", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if encodings:
        avg = np.mean(encodings, axis=0)
        with open("dev_face.pkl", "wb") as f:
            pickle.dump(avg, f)
        print(f"Registered succesfully! {len(encodings)} frames captured.")
    else:
        print("No face detected.")

register_face()