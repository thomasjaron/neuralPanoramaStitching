import torch
import kornia.feature as KF
from kornia.feature.adalam import AdalamFilter

import matplotlib.pyplot as plt

class HomographyEstimator():
    """Class used as an interface for multiple possible keypoint matching and homography estimation methods."""
    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self.matcher = opt.homography_estimation.matcher
        self.feature_extractor = opt.homography_estimation.feature_extractor    
        self.num_features = opt.homography_estimation.num_features

        self.matcher_class = self._get_matcher()
        self.feature_extractor_class = self._get_feature_extractor()
        pass

    def _get_matcher(self):
        # Use the matcher according to the configs loaded into the model
        if self.matcher == 'lightglue':
            matcher_class = KF.LightGlueMatcher("disk").eval().to(self.device)
        else:
            raise NotImplementedError(f"Matcher {self.matcher} not implemented.")
        return matcher_class

    def _get_feature_extractor(self):
        # Use the feature extraction method according to the configs loaded into the model
        if self.feature_extractor == 'disk-depth':
            feature_extractor_class = KF.DISK.from_pretrained("depth", device=self.device)
        elif self.feature_extractor == 'disk-epipolar':
            feature_extractor_class = KF.DISK.from_pretrained("epipolar", device=self.device)
            raise NotImplementedError(f"Feature extractor {self.feature_extractor} not implemented.")
        return feature_extractor_class

    def get_features(self, img1, img2, pad_if_not_divisible=True):
        inp = torch.cat([img1, img2], dim=0)
        features1, features2 = self.feature_extractor_class(inp, self.num_features, pad_if_not_divisible=pad_if_not_divisible)
        # NOTE: Make sure to always return features as a dict containing keypoints and descriptors
        return features1, features2

    def get_matches(self, features1, features2, hw1, hw2):
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors
        lafs1 = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=self.device))
        lafs2 = KF.laf_from_center_scale_ori(kps2[None], torch.ones(1, len(kps2), 1, 1, device=self.device))
        # NOTE: Make sure to always return matches as a dict containing keypoints and descriptors
        return self.matcher_class(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2)
