"""This file contains classes generating our datasets. It is either loading images and their ground truth homographies, or is generating them here."""


import torch
import os
import cv2

import lightning as pl
import numpy as np
import kornia as K

from itertools import combinations
from estimators import HomographyEstimator
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.io import read_image, ImageReadMode

class CustomImageLoader(pl.LightningDataModule):
    """Loads images into the dataloader"""
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.img_dir = opts.img_dir
        if self.opts.use_sidar:
            self.dataset_images = [0] + opts.dataset_images # array of image indices to consider for learning, 0 equals ground truth image
        else:
            self.dataset_images = opts.dataset_images
        # TODO: Dataset images is a list of image names, not indices.
        self.train_dataset = Subset(ImageDataset(self.img_dir), self.dataset_images)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=len(self.dataset_images), shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

class ImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        # Get all image files in the directory
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, [i for i in self.image_files if i.startswith(f'{idx}.')][0])
        image = read_image(img_path, mode=ImageReadMode.RGB).float() / 255.0  # Normalize to [0,1]
            
        return image

class Dataset():
    """Loads images into the dataloader"""
    def __init__(self, opts):
        super().__init__()
        # Presets
        self.opts = opts
        self.device = opts.device
        # Required
        self.hom_dir = opts.hom_dir
        self.dataset_images = opts.dataset_images # array of image indices to consider for learning
        self.rescale_factor = opts.rescale_factor
        # Load full image directory
        self.imgs = None

    def _scale_homographies(self, homs, images):
        """Rescale the images and homographies to the new image size."""
        # homs : [B, 3, 3]
        for i in range(homs.shape[0]):
            # get Image Size for corresponding homography
            img_size = images[i].shape[1:]
            factor_w = img_size[1] / max(img_size[0], img_size[1])
            factor_h = img_size[0] / max(img_size[0], img_size[1])
            S = torch.tensor([
                [factor_w * self.rescale_factor,                              0, 0],
                [                             0, factor_h * self.rescale_factor, 0],
                [                             0,                              0, 1]
            ], dtype=torch.float32, device=self.device)
            S_inv = torch.linalg.inv(S)
            # apply scaling
            for j in range(homs.shape[1]):
                homs[i][j] = S @ homs[i][j] @ S_inv
                det = torch.det(homs[i][j])
                homs[i][j] = homs[i][j] / det.abs().pow(1/3)
        return homs

    def load_sidar_homographies(self, first, selected_indices, used_idxs, input_images, gt_img):
        """Load the homographies from the SIDAR dataset."""
        if not self.opts.use_sidar:
            return torch.eye(3, dtype=torch.float32).repeat(len(selected_indices), len(selected_indices), 1, 1)

        selected_indices = [int(i) for i in selected_indices]
        loaded_homographies = []
        homs_gt_to_x = []
        # Load homographies from gt image (0) to i
        for i in selected_indices:
            invert = True
            fp = os.path.join(self.hom_dir, f"H_0_{i}.mat")
            homography = np.loadtxt(fp)
            homography_tensor = torch.tensor(homography, dtype=torch.float32, device=self.device)
            homography_tensor = torch.linalg.inv(homography_tensor) if invert else homography_tensor
            hw1 = gt_img.shape[1:]
            hw2 = input_images[used_idxs[selected_indices.index(i)]].shape[1:]
            if invert:
                homography_tensor = K.geometry.conversions.normalize_homography(homography_tensor, hw2, hw1).squeeze(0)
            else:
                homography_tensor = K.geometry.conversions.normalize_homography(homography_tensor, hw1, hw2).squeeze(0)
            homs_gt_to_x.append(homography_tensor)

        for i in selected_indices:
            invert = True
            if i == first:
                loaded_homographies.append(homs_gt_to_x[0])
                continue
            # create correct filepath
            fp = os.path.join(self.hom_dir, f"H_{first}_{i}.mat")
            if not os.path.exists(fp):
                # first seems to be higher than i. Load H_i_first instead and invert it.
                fp = os.path.join(self.hom_dir, f"H_{i}_{first}.mat")
                invert = False
            # load homography from file and invert it if necessary
            homography = np.loadtxt(fp)
            homography_tensor = torch.tensor(homography, dtype=torch.float32, device=self.device)
            homography_tensor = torch.linalg.inv(homography_tensor) if invert else homography_tensor
            # hw2, hw1
            hw1 = input_images[used_idxs[selected_indices.index(first)]].shape[1:]
            hw2 = input_images[used_idxs[selected_indices.index(i)]].shape[1:]
            if invert:
                homography_tensor = K.geometry.conversions.normalize_homography(homography_tensor, hw2, hw1).squeeze(0)
            else:
                homography_tensor = K.geometry.conversions.normalize_homography(homography_tensor, hw1, hw2).squeeze(0)
            homography_tensor = torch.matmul(homs_gt_to_x[0], homography_tensor)
            loaded_homographies.append(homography_tensor)
        gt_hom = torch.stack(loaded_homographies)
        gt_hom = self._scale_homographies(gt_hom.unsqueeze(0), input_images).squeeze(0)
        return gt_hom
    
    def _resize_image(self, image):
        """Resize the image by the rescale factor."""
        # First, get image dimensions
        H, W = image.shape[1:]
        # Then, resize the image
        resize = transforms.Compose([
            transforms.Resize((int(H * self.rescale_factor), int(W * self.rescale_factor))),
        ])
        return resize(image)

    def setup_dataset(self, input_images, gt_img=None):
        # Pick images from dataset
        # [1, 3, 4, 5]
        sidar_idxs = self.dataset_images
        first = False

        if not self.opts.estimate_homs:
            # Just return the picked images with identity homographies
            homographies = torch.eye(3, device=self.device).repeat(5, 5, 1, 1)
            first = int(self.dataset_images[0])
        else:
            # Estimate homographies between chosen images, then filter out invalid homographies and return
            # valid dataset elements.
            homography_estimator = HomographyEstimator(self.opts)
            homographies, used_idxs, keypoints = self.calculate_homographies(input_images, homography_estimator)
            imgs = []
            kps = []
            for i in used_idxs:
                if first is False:
                    first = int(self.dataset_images[i])
                imgs.append(input_images[i])
                kps.append(keypoints[used_idxs.index(i)])
            homographies = self._scale_homographies(homographies, input_images)
            # --- Set opts.H and opts.W to match loaded image dimensions before resizing ---
            if imgs and len(imgs) > 0:
                _, H, W = imgs[0].shape
                self.opts.H = H
                self.opts.W = W
                self.opts.output_H = H
                self.opts.output_W = W
            images = torch.stack([self._resize_image(i) for i in imgs])
            sidar_idxs = [self.dataset_images[u] for u in used_idxs]
            # Clean Up
            del homography_estimator
        # Load homographies from SIDAR dataset
        if isinstance(first, int):
            gt_homographies = self.load_sidar_homographies(first, sidar_idxs, used_idxs, input_images, gt_img=gt_img)
            homographies[0] = gt_homographies[0]
        else:
            gt_homographies = None
        print('\nDataset Setup Complete!')
        return images, homographies, gt_homographies, keypoints


    def create_startvalue_homographies(self, homographies, selected_images, keypoints):
        """Create a start value for the homographies based on the images.
        This function assumes that the used images are all connected to each other,
        i.e. the first image eventually has homographies to all other images."""
        dim = len(selected_images)
        homographies_empty = []
        def hom_matrix_is_invalid(hom):
            """Defines the condition on which we consider a homography invalid"""
            # if homography is zero-like matrix, ignore.
            return torch.all(torch.abs(hom) < 0.00001).item()

        ########## 1. Compute missing homographies
        for i in range(dim):
            current_elem_has_hom = False
            for j in range(dim):
                if i == j:
                    continue
                if not hom_matrix_is_invalid(homographies[i][j]):
                    # The image is valid because it connects to another image
                    current_elem_has_hom = True
                    # no new matrix needs to be computed
                else:
                    # if a matrix is missing, try to compute it by using other images that point towards it.
                    for k in range(dim):
                        if k == j:
                            continue
                        if not hom_matrix_is_invalid(homographies[i][k]):
                            if not hom_matrix_is_invalid(homographies[k][j]):
                                if self.opts.debug:
                                    print(f'Transitive Homography computation for row: {i}, col: {j}, k: {k}')
                                # We can compute a homography and its inverse by transitively using a non-first image that points to our image.
                                homographies[i][j] = torch.matmul(homographies[k][j], homographies[i][k])
                                homographies[j][i] = torch.inverse(homographies[i][j])
                                current_elem_has_hom = True
                                break
                            elif not hom_matrix_is_invalid(homographies[j][k]):
                                if self.opts.debug:
                                    print(f'Transitive Homography computation for row: {i}, col: {j}, k: {k}')
                                # We can compute a homography and its inverse by transitively using a non-first image that points to our image.
                                homographies[i][j] = torch.matmul(torch.inverse(homographies[j][k]), homographies[i][k])
                                homographies[j][i] = torch.inverse(homographies[i][j])
                                current_elem_has_hom = True
                                break
            if not current_elem_has_hom:
                homographies_empty.append(i)

        ########## 2. Remove empty rows and columns
        row_mask = ~torch.isin(torch.arange(homographies.size(0)), torch.tensor(homographies_empty))
        col_mask = ~torch.isin(torch.arange(homographies.size(1)), torch.tensor(homographies_empty))
        homographies = homographies[row_mask][:, col_mask]
        dim = homographies.size(0)  # Update dim to match new size of homographies

        def cleanup_keypoints(matrix, empty_indices):
            # Convert empty_indices to a set for fast lookup
            empty_set = set(empty_indices)
            # Remove rows
            filtered_rows = [row for i, row in enumerate(matrix) if i not in empty_set]
            # Remove columns
            filtered_matrix = [[val for j, val in enumerate(row) if j not in empty_set] for row in filtered_rows]
            # turn each column element of each row into a tensor
            filtered_matrix = [[torch.tensor(val) for val in row] for row in filtered_matrix]
            return filtered_matrix


        keypoints = cleanup_keypoints(keypoints, homographies_empty)

        # remove all rows and columns for indexes in homographies[0][i], where that homography is invalid
        for i in range(dim-1, -1, -1):  # Iterate backwards
            if hom_matrix_is_invalid(homographies[0][i]):
                # Remove i'th row and column from homographies using PyTorch operations
                mask = torch.ones(dim, dtype=torch.bool, device=homographies.device)
                mask[i] = False
                homographies = homographies[mask][:, mask]
                # Remove i'th element from keypoints and selected_images
                keypoints = [keypoints[j] for j in range(dim) if j != i]
                selected_images = [selected_images[j] for j in range(dim) if j != i]
                dim = len(selected_images)  # Update dimension to match new length

        if dim < 2:
            raise ValueError("Not enough valid homographies found. Cannot align anything. Retry with a better dataset configuration.")
        ########## 4. Return values
        used_images = [x for x in selected_images if x not in homographies_empty]
        return homographies, used_images, keypoints
    
    def _compute_homography(self, homography_estimator, img1, img2):
        """Takes two images and computes a homography between them using keypoints"""
        hw1 = img1.shape[2:]
        hw2 = img2.shape[2:]

        with torch.inference_mode():
            features1, features2 = homography_estimator.get_features(img1, img2)
            _, idxs = homography_estimator.get_matches(features1, features2, hw1=hw1, hw2=hw2)

        def get_matching_keypoints(kp1, kp2, idxs):
            mkpts1 = kp1[idxs[:, 0]]
            mkpts2 = kp2[idxs[:, 1]]
            return mkpts1, mkpts2
        
        kps1 = features1.keypoints
        kps2 = features2.keypoints
        mkpts1, mkpts2 = get_matching_keypoints(kps1, kps2, idxs)
        assert len(mkpts1) == len(mkpts2)
        keypoint_matches = []
        if len(mkpts1) < 4:
            print("Not enough keypoints found for homography estimation.")
            return torch.zeros((3,3)), [], []
        else:
            hom, inlier_mask = cv2.findHomography(mkpts2.detach().cpu().numpy(), mkpts1.detach().cpu().numpy(), cv2.USAC_MAGSAC, 1.0, 0.999, 100000)
            # normalize the homography
            if hom is None:
                return torch.zeros((3,3)), [], []
            hom = K.geometry.conversions.normalize_homography(torch.tensor(hom), hw2, hw1).squeeze(0)
            mk1 = K.geometry.conversions.normalize_pixel_coordinates(mkpts1, hw1[0], hw1[1]) * self.rescale_factor
            mk2 = K.geometry.conversions.normalize_pixel_coordinates(mkpts2, hw2[0], hw2[1]) * self.rescale_factor
            factor_x1 = hw1[1] / max(hw1[0], hw1[1])
            factor_y1 = hw1[0] / max(hw1[0], hw1[1])
            factor_x2 = hw2[1] / max(hw2[0], hw2[1])
            factor_y2 = hw2[0] / max(hw2[0], hw2[1])
            # multiply each x coordinate by factor_x and each y coordinate by factor_y
            mk1[:, 0] = mk1[:, 0] * factor_x1
            mk1[:, 1] = mk1[:, 1] * factor_y1
            mk2[:, 0] = mk2[:, 0] * factor_x2
            mk2[:, 1] = mk2[:, 1] * factor_y2
            for i in range(len(inlier_mask)):
                if inlier_mask[i]:
                    keypoint_matches.append(torch.stack([mk1[i], mk2[i]]))
        return hom, inlier_mask, torch.stack(keypoint_matches)

    def calculate_homographies(self, selected_imgs, homography_estimator):
        dim = len(selected_imgs)
        # B x B Matrix, where [i,i] = identity, and i,j represents homography from i to j
        homographies = torch.full((dim, dim, 3, 3), float(0)).to(self.device)
        for i in range(dim):

            homographies[i][i] = torch.eye(3, 3, device=self.device)
        # list structure of size dim x dim x (Kpts between i and j) x 2 x 2
        keypoints = [[[] for _ in range(dim)] for _ in range(dim)]

        images_enum = enumerate([sub_tensor for sub_tensor in selected_imgs])
        inliers_all = []
        # Array that saves indiced of finally valid images.
        selected_images = [x for x in range(len(selected_imgs))]
        # index of fixed first image, can change after computation
        # Contains the index of an image having homographies to all other images

        for i1, i2 in combinations(images_enum, 2):
            i, im1 = i1
            j, im2 = i2
            img1 = im1.unsqueeze(0).to(self.device)
            img2 = im2.unsqueeze(0).to(self.device)
            hom, inliers, keypoint_matches = self._compute_homography(homography_estimator, img1, img2)
            if len(inliers) > 0:
                inliers = inliers > 0
                if self.opts.debug:
                    print(f'Inliers between {i} and {j}: {inliers.sum()}')
                # Save results
                inliers_all.append(inliers.sum())
                # Save values from i to j
                homographies[i][j] = hom
                # Save values from j to i
            else:
                homographies[i][j] = torch.zeros((3,3))
                homographies[j][i] = torch.zeros((3,3))
                keypoints[i][j] = []
                if self.opts.debug:
                    print(f'Inliers between {i} and {j}: 0')
                inliers_all.append(0)
            if len(inliers) < 4:
                homographies[j][i] = torch.zeros((3,3))
            else:
                homographies[j][i] = torch.inverse(hom)
            keypoints[i][j] = keypoint_matches

        idx = 0
        for i in range(dim):
            for j in range(i + 1, dim):
                if inliers_all[idx] < self.opts.homography_estimation.min_inliers:
                    # Both directions are invalid
                    if self.opts.debug:
                        print(f'Filtered out homography between {i} and {j} with {inliers_all[idx]} inliers')
                    homographies[i][j] = torch.zeros((3,3))
                    homographies[j][i] = torch.zeros((3,3))
                    keypoints[i][j] = torch.zeros((1, 2, 2)).to(self.device)
                idx += 1

        homographies, used_images, keypoints = self.create_startvalue_homographies(homographies, selected_images, keypoints)
        return homographies, used_images, keypoints
