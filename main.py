# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob
from sort import *

files = glob.glob('output/*.png')
for f in files:
    os.remove(f)

tracker = Sort()
memory = {}
line = [(0, 500), (1920, 500)]  # 在这里可以修改检测线的两点坐标
counter = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default="/home/xwl/code/deep_sort_yolov4/test_video/2record.mp4")
ap.add_argument("-o", "--output", help="path to output video", default="./output/result.avi")
ap.add_argument("-y", "--yolo", help="base path to YOLO directory", default="./yolo-obj")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes) and determine only the *output* layer names
# that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

frameIndex = 0

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities

    # cv2.dnn.blobFromImage 方法参数为:
    # image: 传入的需要进行处理的图像
    # scalefactor: 执行完减均值后，需要缩放图像就需要乘以 scalefactor，默认是 1
    # size: 这是神经网络，真正支持输入的值
    # mean: 这是我们要减去的均值，可以是 R,G,B 均值三元组，或者是一个值，每个通道都减这值
    # swapRB: OpenCV认为图像 通道顺序是B、G、R，而减均值时顺序是R、G、B，为了解决这个矛盾，设置swapRB=True即可
    # yolo 模型在进行训练的时候，图片中的像素点的值都乘以了 1/255 将其压缩到 (0,1) 之间，所以我们在使用 yolo 模型进行检测的时候也要
    # 乘以一个缩放系数
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of the current object detection
            # detection 一共有 85 个维度，分别是 [center_x, center_y, width, height, confidence, class_0_pro, class_1_pro, class_2_pro, class_3_pro ...]
            # 加载的 yolo 模型可以识别 80 个类别的物体，所以后面 80 个维度是我们识别的物体 80 在个 class 上的 confidence
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
                # box 的维度为 (center_x, center_y, width, height)
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences, and class IDs
                # 加入到 boxes 中的 box 的维度为 (left_x, top_y, width, height)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    dets = []

    # 将经过极大值抑制处理的 dets 格式转变为 [left_x, top_y, right_x, bottom_y]
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x + w, y + h, confidences[i]])

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            text = "{}".format(indexIDs[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

    # saves image file
    cv2.imshow("output", frame)
    # Press Q to stop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))

    # write the output frame to disk
    writer.write(frame)

    # increase frame index
    frameIndex += 1

    if frameIndex >= 4000:
        print("[INFO] cleaning up...")
        writer.release()
        vs.release()
        exit()

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
