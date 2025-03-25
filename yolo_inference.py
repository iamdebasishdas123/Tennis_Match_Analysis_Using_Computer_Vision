from ultralytics import YOLO

# Load YOLOv8 model
model= YOLO(r'models\Detect_tennis_ball.pt')
result=model.track('input_videos/input_video.mp4',save=True)

# print(result)

# print("boxes:")

# for box in result[0].boxes:
#     print(box)