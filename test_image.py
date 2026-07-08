from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\chara\OneDrive\Desktop\ANPR_Project\runs\detect\train\weights\best.pt")

img = cv2.imread("car.jpg")

results = model(img, conf=0.6)

annotated = results[0].plot()

cv2.imshow("Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()