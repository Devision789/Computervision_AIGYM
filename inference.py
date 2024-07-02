import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture("data/pullup.mp4")
assert cap.isOpened(), "Error"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("outputs/pullup.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

gym_object = solutions.AIGym(
    line_thickness=2,
    view_img=True,
    pose_type="pullup",
    kpts_to_check=[4, 6, 8],
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video succesfully.")
        break
    results = model.track(im0, verbose=False)  
    
    im0 = gym_object.start_counting(im0, results)
    video_writer.write(im0)

cv2.destroyAllWindows()
video_writer.release()