from model.model_factory.model_builder import get_model
from PIL import Image
from timeit import default_timer as timer
import cv2
import numpy as np


def detect_video(yolo, video_path):
    """
        Helper function to get the object detection up and running
    :param yolo: model object instance containing loaded model
    :param video_path: path for the video source
    :return: None
    """
    
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    count = 0
    
    # Keep iterating till user hits q
    while True:
        # capture image from the device/source
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image, preds = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result,
                    text=fps,
                    org=(3, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50,
                    color=(0, 255, 0),
                    thickness=2)
        print(frame.shape)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1
    yolo.close_session()


if __name__ == '__main__':
    VID_DEVICE = 0
    
    # Build model and get object reference
    yolo_model = get_model('CocoYoloV3')
    
    detect_video(yolo_model, VID_DEVICE)
