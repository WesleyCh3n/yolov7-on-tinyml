import cv2
import numpy as np
import onnxruntime as ort
import time

providers = ["CPUExecutionProvider"]
session = ort.InferenceSession("./best-320.onnx", providers=providers)

img = cv2.imread("./test.jpg").astype(np.float32)  # type: ignore

inputImg = cv2.resize(img / 255.0, (320, 320))  # type: ignore
inputImg = inputImg[:, :, ::-1]
inputImg = inputImg.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
inputImg = np.ascontiguousarray(inputImg)
inputImg = np.expand_dims(inputImg, 0)

outname = [i.name for i in session.get_outputs()]
print(outname)

inname = [i.name for i in session.get_inputs()]
print(inname)

inp = {inname[0]: inputImg}


sum_time = 0
outputs = []
for i in range(60):
    s = time.time()
    outputs = session.run(outname, inp)[0]
    e = time.time()
    sum_time += e-s
    print(f"{e-s}")

print(f"average time: {sum_time/60}")
print(f"fps: {1/(sum_time/60)}")

print(outputs)
