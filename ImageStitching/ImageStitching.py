from OETR.OETRInference import OETRInference
from SuperGluePretrainedNetwork.matching_tool import MatchingTool
from PIL import Image
import numpy as np
import cv2
import copy

class ImageStitching:
    def __init__(self, weight_oetr="OETR/weights/oetr_mf_epoch30_2x4_cyclecenter.pth") -> None:
        self.oetr = OETRInference(weight_path=weight_oetr)
        self.matching = MatchingTool()

    def get_image_stiched_pair(self, img_path_1, img_path_2):
        src_img, src_tensor = self.oetr.process_img(img_path_1)
        dest_img, dest_tensor = self.oetr.process_img(img_path_2)
        bbox1, bbox2 = self.oetr.get_bounding_box(src_tensor, dest_tensor)
        self.oetr.visualize_overlap(src_img, bbox1, dest_img, bbox2, "OETR/test_out/result.png")
        croped_src_img = np.asarray(Image.fromarray(src_img).crop(tuple(bbox1)))
        croped_dest_img = np.asarray(Image.fromarray(dest_img).crop(tuple(bbox2)))
        # cv2.imshow("src crop", croped_src_img)
        # cv2.imshow("src", src_img)
        # cv2.imshow("dest crop", croped_dest_img)
        kp1, kp2 = self.matching.getMatchPoint(croped_src_img, croped_dest_img)
        self.matching.visualizeImages(croped_src_img, croped_dest_img)
        kp1 = np.array([[kp[0]*croped_src_img.shape[1]/self.matching.resize[0], kp[1]*croped_src_img.shape[0]/self.matching.resize[1]] for kp in kp1])
        kp2 = np.array([[kp[0]*croped_dest_img.shape[1]/self.matching.resize[0], kp[1]*croped_dest_img.shape[0]/self.matching.resize[1]] for kp in kp2])
        big_kp1 = np.array([[kp[0]+bbox1[0], kp[1]+bbox1[1]] for kp in kp1])
        big_kp2 = np.array([[kp[0]+bbox2[0], kp[1]+bbox2[1]] for kp in kp2])
        # kp_1_img = copy.deepcopy(croped_src_img)
        # kp_2_img = copy.deepcopy(croped_dest_img)
        # big_kp1_img = copy.deepcopy(src_img)
        # big_kp2_img = copy.deepcopy(dest_img)
        # for e1,e2,e3,e4 in zip(kp1, kp2, big_kp1, big_kp2):
        #     kp_1_img = cv2.circle(kp_1_img, (int(e1[0]),int(e1[1])), radius=3, color=(0, 0, 255), thickness=-1)
        #     kp_2_img = cv2.circle(kp_2_img, (int(e2[0]),int(e2[1])), radius=3, color=(0, 0, 255), thickness=-1)
        #     big_kp1_img = cv2.circle(big_kp1_img, (int(e3[0]),int(e3[1])), radius=3, color=(0, 0, 255), thickness=-1)
        #     big_kp2_img = cv2.circle(big_kp2_img, (int(e4[0]),int(e4[1])), radius=3, color=(0, 0, 255), thickness=-1)
        # cv2.imshow("test",kp_1_img)
        # cv2.imshow("test",kp_2_img)
        # cv2.imshow("test", big_kp1_img)
        # cv2.imshow("test",big_kp2_img)
        (H, status) = cv2.findHomography(big_kp2, big_kp1, cv2.RANSAC)
        print(H)
        h1, w1 = src_img.shape[:2]
        h2, w2 = dest_img.shape[:2]
        result = cv2.warpPerspective(dest_img, H, (w1+w2, h1+h2))
        result[0:h2, 0:w2] = src_img
        # cv2.imshow("test", result)
        # while cv2.waitKey(0) != ord('q'):
        #     break
        return result
