import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio

src_img = imageio.imread(r'C:\Data\Juntion_Hackathon(7000k)\Repo\junctionxhanoi2023\test_img\c.png')
tar_img = imageio.imread(r'C:\Data\Juntion_Hackathon(7000k)\Repo\junctionxhanoi2023\test_img\d.png')
src_gray = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
tar_gray = cv2.cvtColor(tar_img, cv2.COLOR_RGB2GRAY)
# plot_imgs([src_img, tar_img], size=8)

SIFT_detector = cv2.xfeatures2d.SIFT_create()
kp1, des1 = SIFT_detector.detectAndCompute(src_gray, None)
kp2, des2 = SIFT_detector.detectAndCompute(tar_gray, None)

print("KP: ", len(kp1))
print("Des: ", des1.shape)

