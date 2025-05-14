import cv2

# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()