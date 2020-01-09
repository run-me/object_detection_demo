import argparse
from detection.keras_yolo3.yolo import YOLO, detect_video
from PIL import Image


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()


def img_infer(yolo, image):
    r_image, yolo_pred = yolo.detect_image(image)
    # yolo.close_session()
    return r_image, yolo_pred


if __name__ == '__main__':
    vid_device = 0
    yolo_model = YOLO()
    detect_video(yolo_model, vid_device)
