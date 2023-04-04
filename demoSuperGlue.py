from SuperGluePretrainedNetwork.matching_tool import MatchingTool
import cv2
import time
if __name__ == '__main__':
    img1_path = r'SuperGluePretrainedNetwork\imgTest\img1.jpg'
    img2_path = r'SuperGluePretrainedNetwork\imgTest\img2.jpg'
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    a = MatchingTool()
    start = time.time()
    print(a.getMatchPoint(img1,img2))
    end = time.time()
    print(end - start)
    