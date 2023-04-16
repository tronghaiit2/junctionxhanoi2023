from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.superglue import SuperGlue
from SuperGluePretrainedNetwork.models.utils import read_image
import cv2
import time


def main():
    super_point = SuperPoint(
        {
            'nms_radius': 4,
            'keypoint_threshold': 0.00,
            'max_keypoints': -1
        }
    )

    image0, inp0, scales0 = read_image(
        r'C:\Data\Juntion_Hackathon(7000k)\Repo\junctionxhanoi2023\test_img\a.png', "cpu", [640, 480], 0, True)
    image1, inp1, scales1 = read_image(
        r'C:\Data\Juntion_Hackathon(7000k)\Repo\junctionxhanoi2023\test_img\b.png', "cpu", [640, 480], 0, True)

    start = time.perf_counter()
    pred = super_point(
        {
            "image": inp0
        }
    )
    pred_1 = super_point(
        {
            "image": inp1
        }
    )
    end = time.perf_counter()
    print(end - start)
    print(pred)
    exit(0)
    print(type(pred["descriptors"][0]))

    des = pred["descriptors"][0].detach().cpu().numpy()

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


if __name__ == "__main__":
    main()
