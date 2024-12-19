import cv2
import time
from datetime import datetime
from deepface import DeepFace

def recognize_faces(frame):
    """
    Apply face recognition to a video frame and annotate it with results.

    Parameters:
    - frame: The video frame to process.

    Returns:
    - Annotated frame with face recognition results.
    """
    try:
        # Detect faces in the frame
        faces = DeepFace.extract_faces(
            img_path=frame,
            enforce_detection=False
        )

        # Process detected faces
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

                    # No match found in the database
                    if results[0].empty or len(results[0]) == 0:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "INTRU", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    else:
                        # Match found
                        match = results[0].iloc[0]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        label = f"{match['identity'].split('/')[1].split('.')[0].capitalize()}"
                        cv2.putText(frame, label, (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as find_err:
                    print(f"Face matching error: {find_err}")

    except Exception as e:
        print(f"Face detection error: {e}")

    return frame

def save_video_from_stream_with_face_recognition(camera_url, duration=30, output_dir="./"):
    """
    Save a video from a camera stream for a specified duration with face recognition applied to each frame.

    Parameters:
    - camera_url: str, the URL of the camera stream.
    - duration: int, the duration of the video in seconds (default is 30 seconds).
    - output_dir: str, directory where the video file will be saved (default is current directory).
    """
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Get the video properties (fallback to default if unavailable)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 5)  # Frames per second
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for the video file

    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}video_{timestamp}.avi"

    # Create a VideoWriter object
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"Saving video to {output_file}...")

    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from stream.")
            break

        # Apply face recognition to the frame
        frame_with_recognition = recognize_faces(frame)

        # Write the processed frame to the output file
        out.write(frame_with_recognition)

        # Display the frame (optional)
        cv2.imshow("Recording with Face Recognition", frame_with_recognition)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop recording early
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video saved successfully: {output_file}")

# Example usage:
camera_url = "http://192.168.2.200:81/stream"
save_video_from_stream_with_face_recognition(camera_url)
