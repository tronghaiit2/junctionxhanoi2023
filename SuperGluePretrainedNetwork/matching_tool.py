from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import os
import cv2

from .models.matching import Matching
from .models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)

torch.set_grad_enabled(False)


class MatchingTool:
    def __init__(self, resize_float=True, eval=True, input_pairs="assets/scannet_sample_pairs_with_gt.txt", input_dir="assets/scannet_sample_images/", output_dir="assets/dump_match_pairs", max_length=-1, resize = [-1], superglue="indoor", max_keypoints=-1, keypoint_threshold=0.005, nms_radius=4, sinkhorn_iterations=20, match_threshold=0.2, viz=True, viz_extension='png', cache=True):
        self.resize_float = resize_float
        self.img1_cv2 = None
        self.img2_cv2 = None
        self.input_pairs = input_pairs
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_length = max_length
        self.resize = resize
        self.superglue = superglue
        self.max_keypoints = max_keypoints
        self.keypoint_threshold = keypoint_threshold
        self.nms_radius = nms_radius
        self.sinkhorn_iterations = sinkhorn_iterations
        self.match_threshold = match_threshold
        self.viz = viz
        self.eval = eval
        self.viz_extension = viz_extension
        self.cache = cache
        self.pairs = [['img1.png', 'img2.png']]

    def getMatchPoint(self, img1, img2):
        self.img1_cv2 = img1
        self.img2_cv2 = img2
        parent_directory = "."
        directory = r'SuperGluePretrainedNetwork\\assets\\scannet_sample_images'
        os.chdir(directory)
        cv2.imwrite('img1.png', img1)
        cv2.imwrite('img2.png', img2)
        print("img1 size", img1.shape)

        if len(self.resize) == 2 and self.resize[1] == -1:
            self.resize = self.resize[0:1]
        if len(self.resize) == 2:
            pass
        elif len(self.resize) == 1 and self.resize[0] > 0:
            pass
        elif len(self.resize) == 1:
            pass
        else:
            raise ValueError(
                'Cannot specify more than two integers for --resize')

        # Load the SuperPoint and SuperGlue models.
        device = 'cuda' if torch.cuda.is_available() and not self.force_cpu else 'cpu'
        # print('Running inference on device \"{}\"'.format(device))
        config = {
            'superpoint': {
                'nms_radius': self.nms_radius,
                'keypoint_threshold': self.keypoint_threshold,
                'max_keypoints': self.max_keypoints
            },
            'superglue': {
                'weights': self.superglue,
                'sinkhorn_iterations': self.sinkhorn_iterations,
                'match_threshold': self.match_threshold,
            }
        }
        matching = Matching(config).eval().to(device)
        input_dir = Path(self.input_dir)
        output_dir = Path(self.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        timer = AverageTimer(newline=True)

        for i, pair in enumerate(self.pairs):
            name0, name1 = pair[:2]
            stem0, stem1 = Path(name0).stem, Path(name1).stem
            matches_path = output_dir / \
                '{}_{}_matches.npz'.format(stem0, stem1)
            eval_path = output_dir / \
                '{}_{}_evaluation.npz'.format(stem0, stem1)
            viz_path = output_dir / \
                '{}_{}_matches.{}'.format(stem0, stem1, self.viz_extension)
            viz_eval_path = output_dir / \
                '{}_{}_evaluation.{}'.format(stem0, stem1, self.viz_extension)

            # Handle --cache logic.
            do_match = True
            do_eval = self.eval
            do_viz = self.viz
            do_viz_eval = self.eval and self.viz
            if self.cache:
                if matches_path.exists():
                    try:
                        results = np.load(matches_path)
                    except:
                        raise IOError('Cannot load matches .npz file: %s' %
                                      matches_path)

                    kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                    matches, conf = results['matches'], results['match_confidence']
                    do_match = False
                if self.eval and eval_path.exists():
                    try:
                        results = np.load(eval_path)
                    except:
                        raise IOError(
                            'Cannot load eval .npz file: %s' % eval_path)
                    do_eval = False
                if self.viz and viz_path.exists():
                    do_viz = False
                if self.viz and self.eval and viz_eval_path.exists():
                    do_viz_eval = False
                timer.update('load_cache')

            if not (do_match or do_eval or do_viz or do_viz_eval):
                timer.print('Finished pair {:5} of {:5}'.format(
                    i, len(self.pairs)))
                continue

            # If a rotation integer is provided (e.g. from EXIF data), use it:
            # Load the image pair.

            image0, inp0, scales0 = read_image(
                name0, device, self.resize, 0, self.resize_float)
            image1, inp1, scales1 = read_image(
                name1, device, self.resize, 0, self.resize_float)

            if image0 is None or image1 is None:
                print('Problem reading image pair: {} {}'.format(
                    input_dir/name0, input_dir/name1))
                exit(1)
            timer.update('load_image')

            if do_match:
                # Perform the matching.

                pred = matching({'image0': inp0, 'image1': inp1})

                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

                kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                matches, conf = pred['matches0'], pred['matching_scores0']
                timer.update('matcher')

                # Write the matches to disk.
                out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                               'matches': matches, 'match_confidence': conf}
                np.savez(str(matches_path), **out_matches)

            # Keep the matching keypoints.
            valid = matches > -1

            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
        os.chdir(parent_directory)
        return mkpts0, mkpts1

    def visualizeImages(self, img1, img2):
        self.img1_cv2 = img1
        self.img2_cv2 = img2
        directory = r'SuperGluePretrainedNetwork\\assets\\scannet_sample_images'
        os.chdir(directory)
        cv2.imwrite('img1.png', img1)
        cv2.imwrite('img2.png', img2)

        if len(self.resize) == 2 and self.resize[1] == -1:
            self.resize = self.resize[0:1]
        if len(self.resize) == 2:
            pass
        elif len(self.resize) == 1 and self.resize[0] > 0:
            pass
        elif len(self.resize) == 1:
            pass
        else:
            raise ValueError(
                'Cannot specify more than two integers for --resize')

        # Load the SuperPoint and SuperGlue models.
        device = 'cuda' if torch.cuda.is_available() and not self.force_cpu else 'cpu'
        # print('Running inference on device \"{}\"'.format(device))
        config = {
            'superpoint': {
                'nms_radius': self.nms_radius,
                'keypoint_threshold': self.keypoint_threshold,
                'max_keypoints': self.max_keypoints
            },
            'superglue': {
                'weights': self.superglue,
                'sinkhorn_iterations': self.sinkhorn_iterations,
                'match_threshold': self.match_threshold,
            }
        }
        matching = Matching(config).eval().to(device)
        input_dir = Path(self.input_dir)
        output_dir = Path(self.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        timer = AverageTimer(newline=True)

        for i, pair in enumerate(self.pairs):
            name0, name1 = pair[:2]
            stem0, stem1 = Path(name0).stem, Path(name1).stem
            matches_path = output_dir / \
                '{}_{}_matches.npz'.format(stem0, stem1)
            eval_path = output_dir / \
                '{}_{}_evaluation.npz'.format(stem0, stem1)
            viz_path = output_dir / \
                '{}_{}_matches.{}'.format(stem0, stem1, self.viz_extension)
            viz_eval_path = output_dir / \
                '{}_{}_evaluation.{}'.format(stem0, stem1, self.viz_extension)

            # Handle --cache logic.
            do_match = True
            do_eval = self.eval
            do_viz = self.viz
            do_viz_eval = self.eval and self.viz
            if self.cache:
                if matches_path.exists():
                    try:
                        results = np.load(matches_path)
                    except:
                        raise IOError('Cannot load matches .npz file: %s' %
                                      matches_path)

                    kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                    matches, conf = results['matches'], results['match_confidence']
                    do_match = False
                if self.eval and eval_path.exists():
                    try:
                        results = np.load(eval_path)
                    except:
                        raise IOError(
                            'Cannot load eval .npz file: %s' % eval_path)
                    do_eval = False
                if self.viz and viz_path.exists():
                    do_viz = False
                if self.viz and self.eval and viz_eval_path.exists():
                    do_viz_eval = False
                timer.update('load_cache')

            if not (do_match or do_eval or do_viz or do_viz_eval):
                timer.print('Finished pair {:5} of {:5}'.format(
                    i, len(self.pairs)))
                continue

            # If a rotation integer is provided (e.g. from EXIF data), use it:
            # Load the image pair.

            image0, inp0, scales0 = read_image(
                name0, device, self.resize, 0, self.resize_float)
            image1, inp1, scales1 = read_image(
                name1, device, self.resize, 0, self.resize_float)

            if image0 is None or image1 is None:
                print('Problem reading image pair: {} {}'.format(
                    input_dir/name0, input_dir/name1))
                exit(1)
            timer.update('load_image')

            if do_match:
                # Perform the matching.

                pred = matching({'image0': inp0, 'image1': inp1})

                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

                kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                matches, conf = pred['matches0'], pred['matching_scores0']
                timer.update('matcher')

                # Write the matches to disk.
                out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                               'matches': matches, 'match_confidence': conf}
                np.savez(str(matches_path), **out_matches)

            # Keep the matching keypoints.
            valid = matches > -1

            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]
            if do_viz:
            # Visualize the matches.
                color = cm.jet(mconf)
                text = [
                    'SuperGlue',
                    'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                    'Matches: {}'.format(len(mkpts0)),
                ]
            

                # Display extra parameter info.
                k_thresh = matching.superpoint.config['keypoint_threshold']
                m_thresh = matching.superglue.config['match_threshold']
                small_text = [
                    'Keypoint Threshold: {:.4f}'.format(k_thresh),
                    'Match Threshold: {:.2f}'.format(m_thresh),
                    'Image Pair: {}:{}'.format(stem0, stem1),
                ]

                make_matching_plot(
                    image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                    text, viz_path, 1,
                    1, 1, 'Matches', small_text)

                timer.update('viz_match')
