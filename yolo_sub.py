import os
import rclpy
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

MODEL_PATH = os.path.expanduser(
    '~/ros2_ws/src/crazy-taxi/crazytaxi/'
    'newtrain-20261127T155256Z-1-001/newtrain/weights/best.pt')
SAVE_DIR = os.path.expanduser('~/detections')

class YoloSub(Node):
    def __init__(self):
        super().__init__('yolo_sub')
        self.bridge = CvBridge()
        self.model = YOLO(MODEL_PATH)
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.count = 0
        self.create_subscription(Image, '/camera/image_raw',
                                 self.image_callback, 1)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        result = self.model(frame, imgsz=320, conf=0.4, verbose=False)[0]
        labels = [self.model.names[int(c)] for c in result.boxes.cls]

        if labels:
            self.get_logger().info(f'Detected: {labels}')
            self.count += 1
            if self.count % 10 == 0:
                cv2.imwrite(f'{SAVE_DIR}/det_{self.count}.jpg', result.plot())

        if 'human' in labels:
            pass  # TODO: stop
        elif 'bus' in labels or 'semi' in labels:
            pass  # TODO: slow/turn

def main():
    rclpy.init()
    node = YoloSub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()