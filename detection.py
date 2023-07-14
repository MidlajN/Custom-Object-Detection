import cv2
import torch

def detection(weight_path):
    cap = cv2.VideoCapture(0)
    model_path = 'yolov5/runs/train/' + weight_path
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = model(frame).xyxy[0]
        print(detections)
        for detection in detections:
            cv2.rectangle(frame, (int(detection[0]), int(detection[1])), (int(detection[2]), int(detection[3])), (255, 0, 0), 2)
            cv2.putText(frame, detection[4], (int(detection[0]), int(detection[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    weight_path = input('ENTER THE WEIGHT PATH : /yolov5/runs/train/')
    detection(weight_path)