from OETR.OETRInference import OETRInference
from SuperGluePretrainedNetwork.matching_tool import MatchingTool
from PIL import Image
import numpy as np
import cv2

class ImageStitching:
    def __init__(self, weight_oetr="OETR/weights/oetr_mf_epoch30_2x4_cyclecenter.pth") -> None:
        self.oetr = OETRInference(weight_path=weight_oetr)
        self.matching = MatchingTool()

    def get_image_stiched_pair(self, img_path_1, img_path_2):
        src_img, src_tensor = self.oetr.process_img(img_path_1)
        dest_img, dest_tensor = self.oetr.process_img(img_path_2)
        bbox1, bbox2 = self.oetr.get_bounding_box(src_tensor, dest_tensor)
        # self.oetr.visualize_overlap(src_img, bbox1, dest_img, bbox2, "OETR/test_out/result.png")
        croped_src_img = np.asarray(Image.fromarray(src_img).crop(tuple(bbox1)))
        croped_dest_img = np.asarray(Image.fromarray(dest_img).crop(tuple(bbox2)))
        # cv2.imshow("src crop", croped_src_img)
        # cv2.imshow("src", src_img)
        # cv2.imshow("dest crop", croped_dest_img)
        kp1, kp2 = self.matching.getMatchPoint(croped_src_img, croped_dest_img)
        kp_1_img = croped_src_img
        print(len(kp1))
        print(kp2)
        print(np.max(kp1))
        print(croped_src_img.shape)
        for kp in kp1:
            kp_1_img = cv2.circle(kp_1_img, (kp[0],kp[1]), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.imshow("test",kp_1_img)
        mapped_kp1 = np.array([[kp[0]+bbox1[0], kp[1]+bbox1[1]] for kp in kp1])
        while cv2.waitKey(0) != ord('q'):
            break
        print(bbox1, bbox2)
