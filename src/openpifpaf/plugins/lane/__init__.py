
import openpifpaf

from . import lane_kp

from .lane_kp import LaneKp


def register():
    openpifpaf.DATAMODULES['lane'] = lane_kp.LaneKp
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k30-lane'] = ""
    # openpifpaf.CHECKPOINT_URLS['shufflenetv2k30-animalpose'] = \
    #     "http://github.com/vita-epfl/openpifpaf-torchhub/releases/" \
    #     "download/v0.12.9/shufflenetv2k30-210511-120906-animal.pkl.epoch400"
