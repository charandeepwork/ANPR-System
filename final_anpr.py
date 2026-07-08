from ultralytics import YOLO
import cv2
import pytesseract
import re

# Load trained model (IMPORTANT: update path if train2/train3)
model = YOLO(r"C:\Users\chara\OneDrive\Desktop\ANPR_Project\runs\detect\train-2\weights\best.pt")

# Tesseract path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Clean OCR text
def clean_text(text):
    return re.sub('[^A-Z0-9]', '', text)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

seen = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.7)

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        # Take only best detection
        best_box = boxes[boxes.conf.argmax()]

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

        w = x2 - x1
        h = y2 - y1

        # Filter small/noisy boxes
        if w > 80 and h > 30:
            plate = frame[y1:y2, x1:x2]

            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5,5), 0)
            _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            text = pytesseract.image_to_string(
                gray,
                config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )

            plate_text = clean_text(text)

            if plate_text and plate_text not in seen:
                seen.add(plate_text)
                print("Detected Plate:", plate_text)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, plate_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("ANPR System", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()