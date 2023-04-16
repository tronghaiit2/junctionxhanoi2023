from ImageStitching.ImageStitching import ImageStitching
import cv2
import time

if __name__ == "__main__":
    imageStitching = ImageStitching()
    start = time.time()
    result = imageStitching.get_image_stiched_pair(r"test_img\src_4.jpg", r"test_img\dest_4.jpg")
    end = time.time()
    print(end - start)
    cv2.imshow("result", result)
    if cv2.waitKey(0) != ord("q"):
        exit(0)