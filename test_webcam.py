import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

img = cv2.imread("test04.jpg")


results = model(img, conf=0.25)

boxes = results[0].boxes

for box in boxes:
    cls = int(box.cls[0])

    # assuming class 1 = human
    if cls == 1:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # crop human region
        crop = img[y1:y2, x1:x2]

        # run detection again (weapon focus)
        crop_results = model(crop, conf=0.05, imgsz=960)

        # draw boxes on original image
        for cbox in crop_results[0].boxes:
            cx1, cy1, cx2, cy2 = map(int, cbox.xyxy[0])

            # adjust coordinates back to original
            cv2.rectangle(
                img,
                (x1 + cx1, y1 + cy1),
                (x1 + cx2, y1 + cy2),
                (0, 0, 255),
                2
            )

cv2.imshow("Weapon Focus Detection", img)
cv2.waitKey(0)