
import os

import numpy as np

_CATEGORIES = ['lane']  # only for preprocessing

LANE_KEYPOINTS = [
    'Top_left',         # 1
    'Top_right',        # 2
    'Bottom_left',      # 3
    'Bottom_right'      # 4
]


HFLIP = {
    'Top_left': 'Top_left',
    'Top_right': 'Top_right',
    'Bottom_left': 'Bottom_left',
    'Bottom_right': 'Bottom_right'
}


ALTERNATIVE_NAMES = [
    'Top_left',       # 1
    'Top_right',      # 2
    'Bottom_left',    # 3
    'Bottom_right'    # 4
]


LANE_SKELETON = [
    (0, 0), (0, 0), (0, 0), (0, 0)
]


LANE_SIGMAS = [
    1.0,  # nose
    1.0,  # eyes
    1.0,  # eyes
    1.0,  # ears
]

split, error = divmod(len(LANE_KEYPOINTS), 4)
LANE_SCORE_WEIGHTS = [5.0] * split + [3.0] * split + [1.0] * split + [0.5] * split + [0.1] * error

LANE_CATEGORIES = ['lane']

LANE_POSE = np.array([
    [0.0, 0.0, 2.0],  # 'nose',            # 1
    [0.0, 0.0, 2.0],  # 'left_eye',       # 2
    [0.0, 0.0, 2.0],  # 'right_eye',       # 3
    [0.0, 0.0, 2.0]  # 'left_ear',        # 4
])


assert len(LANE_POSE) == len(LANE_KEYPOINTS) == len(ALTERNATIVE_NAMES) == len(LANE_SIGMAS) \
       == len(LANE_SCORE_WEIGHTS), "dimensions!"


def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    show.KeypointPainter.font_size = 0
    keypoint_painter = show.KeypointPainter()

    ann = Annotation(
        keypoints=LANE_KEYPOINTS, skeleton=LANE_SKELETON, score_weights=LANE_SCORE_WEIGHTS)
    ann.set(pose, np.array(LANE_SIGMAS) * scale)
    os.makedirs('all-images', exist_ok=True)
    draw_ann(ann, filename='all-images/lane_pose.png', keypoint_painter=keypoint_painter)


def print_associations():
    for j1, j2 in LANE_SKELETON:
        print(LANE_SKELETON[j1 - 1], '-', LANE_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    print_associations()
    draw_skeletons(LANE_POSE)
