import torch
import cv2
import pytesseract
import re

# Load pretrained plate detection model
model = torch.hub.load('keremberke/yolov5', 'custom', 'lp_detection.pt', source='github')

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def clean_text(text):
    return re.sub('[^A-Z0-9]', '', text)

cap = cv2.VideoCapture(0)

seen = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)

        plate = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        text = pytesseract.image_to_string(
            gray,
            config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        plate_text = clean_text(text)

        if plate_text and plate_text not in seen:
            seen.add(plate_text)
            print("Detected Plate:", plate_text)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("Plate Detection YOLO", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()