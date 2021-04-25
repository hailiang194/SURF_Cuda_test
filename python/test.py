import cv2
import numpy as np
import time

image = cv2.imread("../image.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

#CPU
surf = cv2.xfeatures2d.SURF_create()
start = time.perf_counter()
kp, desc = surf.detectAndCompute(gray_image, np.array([]))
print("CPU Time:",time.perf_counter() - start)
kp_image = cv2.drawKeypoints(gray_image, kp, np.array([]))

#GPU
gpu_gray_image = cv2.cuda_GpuMat()
gpu_gray_image.upload(gray_image)

print("Enable device:", cv2.cuda.getCudaEnabledDeviceCount())

surf_cuda = cv2.cuda.SURF_CUDA_create(100)

start = time.perf_counter()
gpu_kp_mat, gpu_desc = surf_cuda.detectWithDescriptors(gpu_gray_image, cv2.cuda_GpuMat(np.array([])))
print("GPU Time:", time.perf_counter() - start)

gpu_kp = surf_cuda.downloadKeypoints(gpu_kp_mat)

gpu_kp_image = cv2.drawKeypoints(gray_image, gpu_kp, np.array([]))

cv2.imshow("CPU", kp_image)
cv2.imshow("GPU", gpu_kp_image)
cv2.waitKey(0)
