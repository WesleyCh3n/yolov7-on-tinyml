import os

import cv2
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(
    model_path="./best.tflite",
    num_threads=8
)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data
input_shape = input_details[0]["shape"]
img = cv2.imread("./test.jpg").astype(np.float32)  # type: ignore
img1 = cv2.resize(img / 255.0, input_shape[2:])  # type: ignore
img1 = img1[:, :, ::-1]
img1 = img1.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
img1 = np.ascontiguousarray(img1)

interpreter.set_tensor(input_details[0]["index"], [img1])

s = time.time()
interpreter.invoke()
e = time.time()
print(f"{e-s}")

# get_tensor() returns a copy of the tensor data
# use tensor() in order to get a pointer to the tensor
output_data = interpreter.get_tensor(output_details[0]["index"])

# 7 means batchid,x0,y0,x1,y1,classid,score
select_id = np.where(output_data[:, 6] > 0.4)
select_result = output_data[select_id]
# scale box to original image
select_result[:, 1:5] = (
    select_result[:, 1:5]
    / 640
    * [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
)
for *xyxy, cls, conf in select_result[:, 1:]:
    print(xyxy, cls, conf)
    pt1, pt2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    img = cv2.rectangle(img, pt1, pt2, (0, 255, 255), 1)

cv2.imwrite("./result.jpg", img)
