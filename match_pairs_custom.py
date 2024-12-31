from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import os
import cv2

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
from pyTorchAutoForge.utils import GetDevice

torch.set_grad_enabled(False)


def DefineSuperPointSuperGlueModel():
    """
    DefineSuperPointSuperGlueModel _summary_

    _extended_summary_

    :return: _description_
    :rtype: _type_
    """
    # Tuning settings
    confidence_threshold = 0.7  # Adjust to filter weaker matches
    nms_radius = 25
    keypoint_threshold = 0.1
    max_keypoints = -1
    superglue = 'outdoor'
    sinkhorn_iterations = 35
    match_threshold = 0.35
    resize_value = [512, 512]

    # Get the device
    device = GetDevice()
    print('Running inference on device \"{}\"'.format(device))

    # Load the SuperPoint and SuperGlue models
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }

    model_extractor_matcher = Matching(config).eval().to(device)

    return model_extractor_matcher


def main():
    # Change folder to the one containing the script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    input_dir = Path("input_pairs")
    output_dir = Path("output_matches")
    output_dir_ransac = Path("output_matches_ransac")

    do_viz = True
    do_match = True
    do_ransac_essential = True
    apply_confidence_thr = False
    use_fast_viz = False

    input_pairs_txt = "input_pairs.txt"
    show_keypoints = True
    display_ocv = True

    # Load the SuperPoint and SuperGlue models
    model_extractor_matcher = DefineSuperPointSuperGlueModel()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(output_dir_ransac):
        os.mkdir(output_dir_ransac)
        
    if do_viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))

    with open(input_pairs_txt, 'r') as f:
        pairs = [l.split() for l in f.readlines()]
    
    pairs = pairs[0:len(pairs)]

    timer = AverageTimer(newline=True)
    
    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2] # Get name of images from pair
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        viz_path = output_dir / '{}_{}_matches.png'.format(stem0, stem1)

        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0

        # Load the image pair.
        image0, inp0, scales0 = read_image(
            input_dir / name0, device, resize_value, rot0, True)
        image1, inp1, scales1 = read_image(
            input_dir / name1, device, resize_value, rot1, True)

        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir/name0, input_dir/name1))
            exit(1)
        timer.update('load_image')

        if do_match:
            # Perform the matching.
            pred = model_extractor_matcher({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, match_confidences = pred['matches0'], pred['matching_scores0']
            timer.update('matcher')

            #if apply_confidence_thr:
            #    # Filter matches based on confidence threshold
            #    matches = matches[match_confidences >
            #                           confidence_threshold]
            #    # Extract coordinates of matched keypoints
            #    kpts0 = kpts0[matches > -1]
            #    kpts1 = kpts1[matches]

            # Write the matches to disk.
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': match_confidences}

            np.savez(str(matches_path), **out_matches)

            if do_ransac_essential:

                matches_path_ransac = output_dir_ransac / \
                    '{}_{}_matches_ransac.npz'.format(stem0, stem1)
                viz_path_ransac = output_dir_ransac / \
                    '{}_{}_matches_ransac.png'.format(stem0, stem1)

                # Extract keypoints coordinates from matches
                valid = matches > -1
                points_A = np.float32(kpts0[valid])
                points_B = np.float32(kpts1[matches[valid]])

                # Use RANSAC to estimate fundamental matrix and filter outliers
                F, mask = cv2.findFundamentalMat(points_A, points_B, cv2.FM_RANSAC, ransacReprojThreshold=0.5, confidence=0.99)

                # Select inlier matches based on RANSAC mask
                inlier_matches = [matches[i]for i in range(len(mask)) if mask[i]]
                # Get confidence of inlier matches
                match_confidences_ransac = [match_confidences[i] for i in range(len(mask)) if mask[i]]

                # Get inliers keypoints
                kpts0_ransac = np.array([points_A[i] for i in range(len(mask)) if mask[i]])
                kpts1_ransac = np.array([points_B[i] for i in range(len(mask)) if mask[i]])

                # Write the matches to disk.
                out_matches_inliers = {'keypoints0': kpts0_ransac, 'keypoints1': kpts1_ransac,
                                       'matches': inlier_matches, 'match_confidence': match_confidences_ransac}
                
                np.savez(str(matches_path_ransac), **out_matches_inliers)

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = match_confidences[valid]

        # If ransac is enabled, also keep the ransac inliers
        if do_ransac_essential:
            mkpts0_ransac = kpts0_ransac
            mkpts1_ransac = kpts1_ransac
            mconf_ransac = match_confidences_ransac

        if do_viz:

            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info.
            k_thresh = model_extractor_matcher.superpoint.config['keypoint_threshold']
            m_thresh = model_extractor_matcher.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, show_keypoints,
                use_fast_viz, display_ocv, 'Matches', small_text)

            timer.update('viz_match')

            # If ransac is enabled, also visualize the inliers
            if do_ransac_essential:
                color_ransac = cm.jet(mconf_ransac)
                text_ransac = [
                    'SuperGlue + RANSAC',
                    'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                    'Matches: {}'.format(len(mkpts0_ransac)),
                ]
                if rot0 != 0 or rot1 != 0:
                    text_ransac.append('Rotation: {}:{}'.format(rot0, rot1))

                make_matching_plot(
                    image0, image1, kpts0, kpts1, mkpts0_ransac, mkpts1_ransac, color_ransac,
                    text_ransac, viz_path_ransac, show_keypoints,
                    use_fast_viz, display_ocv, 'Matches RANSAC', small_text)

                timer.update('viz_match_ransac')

        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

if __name__ == '__main__':
    main()