from matching_tool import MatchingTool
import cv2
if __name__ == '__main__':
    img1_path = r'imgTest\img1.jpg'
    img2_path = r'imgTest\img2.jpg'
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    a = MatchingTool()
    print(a.getMatchPoint(img1,img2))
    