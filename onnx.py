import cv2
import numpy as np
import onnxruntime as ort
import time
from threading import Thread

name_lst = ['1', '2', '3', '4', '5', '6', '7', '8']
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# ONNX 模型文件路径
model_path = "best-2.onnx"
session = ort.InferenceSession(model_path)

w = 1920
h = 1080

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = max(line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1, 1)  # 线条/字体粗细
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # 字体粗细
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, tf, thickness=1)[0]
        c2 = c1[0] + t_size[0] + 10, c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # 填充
        cv2.putText(img, label, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_PLAIN, tf, [225, 255, 255], 1)

def infer_image(img0):
    img = cv2.resize(img0, (640, 640))
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, swapRB=True)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    results = session.run([output_name], {input_name: blob})
    return results

def m_detection():
    global det_boxes_show, scores_show, ids_show
    while True:
        ret, img0 = cap.read()
        if not ret:
            print("无法读取帧")
            break

        results = infer_image(img0)

        det_boxes = []
        scores = []
        ids = []

        for data in results[0][0]:
            if data[4] < 0.6:
                continue

            x1, y1, x2, y2, conf, cls = data
            x1 = int(x1 / 640 * w)
            x2 = int(x2 / 640 * w)
            y1 = int(y1 / 640 * h)
            y2 = int(y2 / 640 * h)
            cls = int(cls)

            det_boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            ids.append(cls)

        det_boxes_show = det_boxes
        scores_show = scores
        ids_show = ids

if __name__ == "__main__":
    global det_boxes_show, scores_show, ids_show
    det_boxes_show, scores_show, ids_show = [], [], []

    m_thread = Thread(target=m_detection, daemon=True)
    m_thread.start()

    while True:
        ret, img0 = cap.read()
        if not ret:
            print("无法读取帧")
            break

        for box, score, id in zip(det_boxes_show, scores_show, ids_show):
            label = '%s:%.2f' % (name_lst[id], score)
            plot_one_box(box, img0, color=(255, 0, 0), label=label, line_thickness=2)

        cv2.imshow("video", img0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()