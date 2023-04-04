import cv2
import torch
from .src.model import OETR
from .src.config.default import get_cfg_defaults

torch.set_grad_enabled(False)

class OETRInference:
    def __init__(self, weight_path) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg = get_cfg_defaults()
        self.model = OETR(self.cfg.OETR).eval().to(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device(self.device)))

    def frame2tensor(self,frame):
        return torch.from_numpy(frame / 255.0).float()[None].to(self.device)

    def process_img(self, path_img, size = (640, 480)):
        img = cv2.imread(path_img)
        img = cv2.resize(img, size)
        tensor = self.frame2tensor(img)
        return img, tensor
    
    def visualize_overlap(self, image1, bbox1, image2, bbox2, output):
        left = cv2.rectangle(image1, tuple(bbox1[0:2]), tuple(bbox1[2:]),
                            (255, 0, 0), 2)
        right = cv2.rectangle(image2, tuple(bbox2[0:2]), tuple(bbox2[2:]),
                            (0, 0, 255), 2)
        viz = cv2.hconcat([left, right])
        cv2.imwrite(output, viz)

    def get_bounding_box(self, src_tensor, dest_tensor):
        box1, box2 = self.model.forward_dummy(src_tensor, dest_tensor)
        np_box1 = box1[0].cpu().numpy().astype(int)
        np_box2 = box2[0].cpu().numpy().astype(int)
        return np_box1, np_box2
