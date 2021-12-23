#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re

# percentage of pixels > 1 required to assign a foreground label to a patch
foreground_threshold = 0.25

# assign a label to a patch


def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(full_path, image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(full_path)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, submission_mask_path, image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            path = os.path.join(submission_mask_path, fn)
            f.writelines('{}\n'.format(s)
                         for s in mask_to_submission_strings(path, fn))


def submit():
    submission_filename = 'DinkNet152.csv'
    submission_mask_path = 'submits/DinkNet152/'
    end = int(len('_mask.png'))
    start = int(len('test_'))
    image_names = os.listdir(submission_mask_path)
    image_names.sort(key=lambda x: int(x[start:][:-end]))
    masks_to_submission(submission_filename, submission_mask_path, image_names)
