from model.model_factory.yolo import YOLO


class CocoYoloV3:
    def __init__(self):
        self.MODEL_PATH = '/home/mash/universe/dev/github-runme/object_detection_demo/model/model_data/coco/yolov3' \
                          '-coco.h5'
        self.ANCHOR_PATH = '/home/mash/universe/dev/github-runme/object_detection_demo/model/model_data/anchors' \
                           '/tiny_yolo_anchors.txt'
        self.CLASSES_PATH = '/home/mash/universe/dev/github-runme/object_detection_demo/model/model_data/coco' \
                            '/coco_classes.txt'
        
        self.SCORE = 0.1
        self.IOU = 0.3
        self.MODEL_IMAGE_SIZE = (416, 416)
    
    def build_model(self):
        # construct yolo model here with object parameters
        yolo_instance = YOLO(model_path=self.MODEL_PATH,
                             anchors_path=self.ANCHOR_PATH,
                             classes_path=self.CLASSES_PATH,
                             score=self.SCORE,
                             iou=self.IOU,
                             model_image_size=self.MODEL_IMAGE_SIZE)
        return yolo_instance


def get_model(model_name):
    model_map = {'CocoYoloV3': CocoYoloV3()}
    return model_map[model_name].build_model()
