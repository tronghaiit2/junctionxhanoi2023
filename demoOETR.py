from OETR.OETRInference import OETRInference

def main():
    oetr = OETRInference("OETR/weights/oetr_mf_epoch30_2x4_cyclecenter.pth")
    src_img, src_tensor = oetr.process_img("OETR/test_img/src_2.png")
    dest_img, dest_tensor = oetr.process_img("OETR/test_img/dest_2.png")
    bbox1, bbox2 = oetr.get_bounding_box(src_tensor, dest_tensor)
    print(bbox1, bbox2)
    oetr.visualize_overlap(src_img, bbox1, dest_img, bbox2, "OETR/test_out/result.png")

if __name__ == "__main__":
    main()