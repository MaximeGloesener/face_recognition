import cv2
from deepface import DeepFace

def recognize_faces(frame):
    try:
        # Détection des visages sur la frame
        faces = DeepFace.extract_faces(
            img_path=frame,
            enforce_detection=False
        )

        # On parcourt les visages détectés:
        # - extraction de la région faciale
        # - recherche de correspondance dans la base de données, si visage connu alors on affiche le nom, sinon on affiche "INTRU"
        for face in faces:
            if face and 'facial_area' in face:
                region = face['facial_area']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']


                face_crop = frame[y:y+h, x:x+w]

                try:
                    results = DeepFace.find(
                        img_path=face_crop,
                        db_path="db",
                        model_name='VGG-Face',
                        enforce_detection=False
                    )

                    # Si pas de correspondance trouvée dans la base de données
                    if results[0].empty or len(results[0]) == 0:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "INTRU", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    else:
                        # Si correspondance trouvée dans la base de données
                        match = results[0].iloc[0]
                        print(match)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        label = f"{match['identity'].split('/')[1].split('.')[0].capitalize()}"
                        cv2.putText(frame, label, (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as find_err:
                    print(f"Face matching error: {find_err}")

    except Exception as e:
        print(f"Face detection error: {e}")

    return frame

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        frame_with_recognition = recognize_faces(frame)

        cv2.imshow('Face Recognition', frame_with_recognition)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()