import cv2
import pytesseract
import imutils
import mysql.connector
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# DB connection
db = mysql.connector.connect(
    host="localhost",
    user="anpr_user",
    password="1234",
    database="anpr"
)
cursor = db.cursor()

def clean_text(text):
    return re.sub('[^A-Z0-9]', '', text)

# START CAMERA
cap = cv2.VideoCapture(0)

seen = set()  # avoid duplicate entries

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = imutils.resize(frame, width=600)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(c)
            plate = gray[y:y+h, x:x+w]

            text = pytesseract.image_to_string(plate, config='--psm 8')
            plate_text = clean_text(text)

            if plate_text and plate_text not in seen:
                seen.add(plate_text)

                print("Detected:", plate_text)

                # Save to DB
                query = "INSERT INTO plates (plate_number) VALUES (%s)"
                cursor.execute(query, (plate_text,))
                db.commit()

                cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)

                break

    cv2.imshow("Real-Time ANPR", image)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()