import torch
import numpy as np
import cv2
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator

MODEL_PATH = 'runs/train/exp4/weights/best.pt'
MODEL_PATH2 = 'yolov5s.pt'

img_size = 640
conf_thres = 0.5  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 1000  # maximum detections per image
classes = None  # filter by class
classes2 = [0,1,2,3,5,7]  # filter by class
agnostic_nms = False  # class-agnostic NMS

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device2 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(MODEL_PATH, map_location=device)
ckpt2 = torch.load(MODEL_PATH2, map_location=device2)
model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()
model2 = ckpt2['ema' if ckpt2.get('ema') else 'model'].float().fuse().eval()
class_names = ['횡단보도', '빨간불', '초록불'] # model.names
class_names2 = ['사람','자전거','일반차량','오토바이','버스','트럭'] # model.names
stride = int(model.stride.max())
stride2 = int(model2.stride.max())
colors = ((50, 50, 50), (0, 0, 255), (0, 255, 0)) # (gray, red, green)
colors2 = ((0, 255, 255), (255, 0, 100), (255, 0, 0),(255, 0, 0),(255, 0, 0),(255, 0, 0)) # (yellow, purple, blue)

cap = cv2.VideoCapture('data/sample.mp4')

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('data/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    #  횡단보도 preprocess
    img_input = letterbox(img, img_size, stride=stride)[0]
    img_input = img_input.transpose((2, 0, 1))[::-1]
    img_input = np.ascontiguousarray(img_input)
    img_input = torch.from_numpy(img_input).to(device)
    img_input = img_input.float()
    img_input /= 255.
    img_input = img_input.unsqueeze(0)

    # 횡단보도 inference
    pred = model(img_input, augment=False, visualize=False)[0]

    # 횡단보도 postprocess
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

    pred = pred.cpu().numpy()

    pred[:, :4] = scale_coords(img_input.shape[2:], pred[:, :4], img.shape).round()

    # 사람, 차량 preprocess
    img_input2 = letterbox(img, img_size, stride=stride2)[0]
    img_input2 = img_input2.transpose((2, 0, 1))[::-1]
    img_input2 = np.ascontiguousarray(img_input2)
    img_input2 = torch.from_numpy(img_input2).to(device2)
    img_input2 = img_input2.float()
    img_input2 /= 255.
    img_input2 = img_input2.unsqueeze(0)

    # 사람, 차량 inference
    pred2 = model2(img_input2, augment=False, visualize=False)[0]


    # 사람, 차량 postprocess
    pred2 = non_max_suppression(pred2, conf_thres, iou_thres, classes2, agnostic_nms, max_det=max_det)[0]

    pred2 = pred2.cpu().numpy()

    pred2[:, :4] = scale_coords(img_input2.shape[2:], pred2[:, :4], img.shape).round()


    # Visualize
    annotator = Annotator(img.copy(), line_width=3, example=str(class_names), font='data/malgun.ttf')

    cw_x1, cw_x2 = 0, 0 # 횡단보도 좌측(cw_x1), 우측(cw_x2) 좌표

    # 횡단보도
    for p in pred:
        class_name = class_names[int(p[5])]

        x1, y1, x2, y2 = p[:4]

        annotator.box_label([x1, y1, x2, y2], '%s %d' % (class_name, float(p[4]) * 100), color=colors[int(p[5])])

        if class_name == '횡단보도':
            cw_x1, cw_x2 = x1, x2

    # 사람, 차량
    for p in pred2:
        try:
            class_name = class_names2[int(p[5])]
        except IndexError:
            continue

        x3, y3, x4, y4 = p[:4]

        alert_text = ''
        color = colors2[int(p[5])]

        if class_name == '사람':
            if cw_x1 < x4 < cw_x2 and y1 < y4 < y2:
                alert_text = '사람이 횡단보도를 건너고 있습니다. '
                color = (0, 255, 0) # green

        if class_name == '일반차량' or class_name == '버스' or class_name == '트럭' or class_name == '오토바이' or class_name == '자전거':
            if x4 < cw_x1: # 왼쪽 차량
                alert_text = str(cw_x1 - (x4)) + 'm 왼쪽 방향 '
                color = (255, 0, 0) # blue
            elif x3 > cw_x2: # 오른쪽 차량
                alert_text = str(x3 - cw_x2) + 'm 오른쪽 방향 '
                color = (255, 0, 0) # blue
            elif cw_x1 < x3 < cw_x2 or cw_x1 < x4 < cw_x2:
                alert_text = class_name + '이 횡단보도에 있습니다.'
                color = (0, 0, 255) # red


        annotator.box_label([x3, y3, x4, y4], '%s %d' % (alert_text+class_name, float(p[4]) * 100), color=color)


    result_img = annotator.result()

    cv2.imshow('result', result_img)
    out.write(result_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()