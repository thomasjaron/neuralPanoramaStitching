"""This file contains all network definitions."""
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from easydict import EasyDict as edict
from utils import Utils, Warp

# Code structure adapted from BARF
# https://github.com/chenhsuanlin/bundle-adjusting-NeRF

class HomographyFunction(nn.Module):
    def __init__(self, opt, logger):
        super().__init__()
        self.opt = opt
        self.device = opt.device
        self.neural_image = NeuralImageFunction(opt)
        # represents the homographies for each input image
        self.warp_param = torch.nn.Embedding(opt.batch_size, opt.warp.dof).to(self.device)
        torch.nn.init.zeros_(self.warp_param.weight)
        # save the predictions in here as well
        self.pred = edict()
        Utils.move_to_device(self.pred, self.device)

        # Utility variables
        self.batch_size = opt.batch_size
        self.warp = Warp(opt)
        self.h = self.opt.H * self.opt.rescale_factor
        self.w = self.opt.W * self.opt.rescale_factor
        # Iterations
        self.it = 0
        self.logger = logger

    def initialize_warp_params(self, homographies):
        """Initialize the warp parameters with the given homographies"""
        # transform incoming homographies. [B - 1, 3, 3] -> [B, 8]

        homs = homographies[0]
        self.warp_param.weight.data = self.warp.SL3_homs_to_sl3_stack(homs)
        if self.opt.debug == True:
            print(f'Current Batch Size in Hom Func: {self.batch_size}')
            print("Warp Param shape: ", self.warp_param.weight.data.shape)
            print("Warp Param data: ", self.warp_param.weight.data)
        self.homographies = homographies

    def forward(self, it, mode=None): # pylint: disable=unused-argument
        """Get image and mask predictions given the current estimated homographies"""
        xy_grid = self.warp.get_normalized_pixel_grid(crop=self.opt.use_cropped_images)
        ############ Neural Image Prediction ###########################
        xy_grid_warped = self.warp.warp_grid(xy_grid, self.warp_param.weight)
        # Do not forward or learn the neural image until keypoints are somewhat aligned yet.
        if it / self.opt.max_iter >= self.opt.loss_threshold:
            self.pred.rgb = self.neural_image.forward(xy_grid_warped) # [B, HW, 3]
            self.pred.rgb_map = self.pred.rgb.view(self.batch_size, int(self.h), int(self.w), 3).permute(0, 3, 1, 2) # [B, 3, H, W]
            return self.pred
        else:
            return None
        
    def simple_poisson_loss(self, pred, images):
        """Compute Poisson loss between predicted and ground truth images."""
        # Compute gradient of each image
        pred_grad = self.compute_grad(pred) # [B, 3, 2, H, W]
        images_grad = self.compute_grad(images) # [B, 3, 2, H, W]
        # Compute Poisson loss
        loss = self.l1_loss(pred_grad, images_grad)
        return loss
        
    def compute_loss(self, images, kps, it, mode=None): # pylint: disable=unused-argument
        """Compute loss value"""
        loss = edict()
        a, b, c = self.get_weights(it, self.opt.max_iter)
        loss.rgb = self.mse_loss(self.pred.rgb_map, images) if a else 0
        loss.kps = self.keypoints_loss(kps) if b else 0
        loss.grad = self.pyramid_grad_loss(self.pred.rgb_map, images, it) if c else 0
        loss.render = a * loss.rgb + b * loss.kps + c * loss.grad
        self.log_metrics(loss, it, a)
        return loss
        
    def keypoints_loss(self, kps):
        homs = self.warp.sl3_to_SL3(self.warp_param.weight)  # [B, 3, 3]
        loss = torch.tensor(0.0, device=self.device)
        for i in range(len(kps)): 
            for j in range(i + 1, len(kps)):
                kps_ij = kps[i][j]  # [num_pairs, 2, 2]
                if kps_ij.shape[0] < self.opt.homography_estimation.min_inliers:
                    continue
                kps_i = kps_ij[:, 0, :]
                kps_j = kps_ij[:, 1, :]
                ones = torch.ones(*kps_i.shape[:-1], 1, device=self.device)
                kps_i = torch.cat([kps_i, ones], dim=-1)  # [num_pairs, kp_num, 3]
                kps_j = torch.cat([kps_j, ones], dim=-1)  # [num_pairs, kp_num, 3]
                
                # Transform keypoints
                kps_i = (kps_i @ homs[i].transpose(-1, -2))  # [num_pairs, kp_num, 3]
                kps_j = (kps_j @ homs[j].transpose(-1, -2))  # [num_pairs, kp_num, 3]
                # Normalize homogeneous coordinates
                kps_i = kps_i / kps_i[..., 2:3]
                kps_j = kps_j / kps_j[..., 2:3]
                
                # Compute mean absolute difference, weighted by the number of pairs
                loss += self.l1_loss(kps_i, kps_j) * kps_ij.shape[0]
        loss = loss
        return loss

    def mse_loss(self, pred, labels):
        """Perform MSE on prediction and groundtruth images and use masks if available."""
        loss = (pred.contiguous() - labels) ** 2
        loss = loss.mean()
        return loss
    
    def l1_loss(self, pred, labels):
        loss = torch.abs(pred.contiguous() - labels)
        loss = loss.mean()
        return loss

    def pyramid_grad_loss(self, pred, target, it):
        """Compute gradient loss at multiple scales."""
        B, C, H, W = pred.shape
        levels = self.opt.blending_depth
        resize_factor = self.opt.blending_resize_factor
        
        total_loss = 0

        for i in range(len(pred)):
            # Build gaussian pyramids for both images.
            pred_pyramid = self.build_pyramid(pred[i].unsqueeze(0), levels, resize_factor)
            target_pyramid = self.build_pyramid(target[i].unsqueeze(0), levels, resize_factor)

            # Build laplacian pyramids for both images.
            pred_laplacian = self.build_laplacian_pyramid(pred_pyramid, levels)
            target_laplacian = self.build_laplacian_pyramid(target_pyramid, levels)

            # # Build gradient pyramids for both images.
            pred_grad = self.build_gradient_pyramid(pred[i].unsqueeze(0), levels, resize_factor)
            target_grad = self.build_gradient_pyramid(target[i].unsqueeze(0), levels, resize_factor)

            if levels == 0:
                # Special case, only directly compare the fullsize gradients/magnitudes.
                total_loss += self.l1_loss(pred_grad[0], target_grad[0])
            else:
                # Compare the laplacian pyramids.
                for j in range(levels - 1):
                    total_loss += self.l1_loss(pred_laplacian[j], target_laplacian[j])
                # Compare the gradient pyramids.
                for j in range(levels):
                    total_loss += self.l1_loss(pred_grad[j], target_grad[j])

        return total_loss

    def build_pyramid(self, img, levels, resize_factor, use_gauss_conv=True):
        # img: [C, H, W]
        pyramid = [img]
        # Define 3x3 Gaussian kernel and normalize
        base_kernel = torch.tensor([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], dtype=torch.float32, device=self.device)
        base_kernel /= base_kernel.sum()

        # Expand kernel for grouped convolution
        C = img.shape[1]  # Channel count (typically 3)
        gauss_kernel = base_kernel.view(1, 1, 3, 3).expand(C, 1, 3, 3)
        for _ in range(1, levels):
            i = pyramid[-1]
            if use_gauss_conv:
                i = F.conv2d(i, gauss_kernel, padding=0, groups=3)  # (C, H, W)
            i = F.interpolate(i, scale_factor=resize_factor, mode='bilinear', align_corners=False, recompute_scale_factor=False)
            pyramid.append(i)
        return pyramid

    def build_laplacian_pyramid(self, gaussian_pyramid, levels):
        laplacian_pyramid = []
        for i in range(levels - 1):
            upsampled = F.interpolate(gaussian_pyramid[i + 1], size=gaussian_pyramid[i].shape[-2:], mode='bilinear', align_corners=False)
            lap = gaussian_pyramid[i] - upsampled
            laplacian_pyramid.append(lap)
        return laplacian_pyramid
    
    def build_gradient_pyramid(self, image, levels, resize_factor):
        gradient_pyramid = []
        pyramid = self.build_pyramid(image, levels, resize_factor, use_gauss_conv=False)
        if levels == 0:
            levels = 1
            resize_factor = 1
        for i in range(levels):
            grad = F.interpolate(pyramid[i], scale_factor=resize_factor, mode='bilinear', align_corners=False, recompute_scale_factor=False)
            grad = self.compute_grad(grad)
            gradient_pyramid.append(grad)
        return gradient_pyramid

    def compute_grad(self, inr):
        # inr: [B, 3, H, W]
        B, C, H, W = inr.shape  # Extract dimensions
        # Define Sobel kernels
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.device)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.device)

        # Expand kernels to match input channels (C) and batch processing
        sobel_x_kernel = sobel_x_kernel.expand(C, 1, 3, 3)  # (C, 1, 3, 3)
        sobel_y_kernel = sobel_y_kernel.expand(C, 1, 3, 3)  # (C, 1, 3, 3)

        # Apply convolution to each channel independently
        grad_x = F.conv2d(inr, sobel_x_kernel, padding=0, groups=C)  # (B, C, H, W)
        grad_y = F.conv2d(inr, sobel_y_kernel, padding=0, groups=C)  # (B, C, H, W)

        # Stack gradients to get shape (B, C, 2, H, W), where 2 corresponds to (dx, dy)
        gradients = torch.stack((grad_x, grad_y), dim=2)  # Shape: (B, C, 2, H, W)

        return gradients
    
    def get_weights(self, it, max_iter):
        """Get weights based on current iteration."""
        weights = self.opt.loss_weights
        # Activate blending and color loss only after threshold passed
        if weights[2] == 1 and it / max_iter < self.opt.loss_threshold:
            return (0, weights[1], 0)
        return weights

    def log_metrics(self, loss, it, a):
        # log quantitative metrics, if rgb loss is active
        if self.opt.tb and it % self.opt.log_scalar == 0 or it == self.opt.max_iter - 1:
            if a > 0:
                # RMSE
                self.logger.experiment.add_scalar("Metrics/RMSE", loss.rgb.sqrt(), it)
                # PSNR
                self.logger.experiment.add_scalar("Metrics/PSNR", -10 * loss.rgb.log10(), it)

class NeuralImageFunction(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.device = opt.device
        # positional encoding related settings
        self.posenc = opt.posenc # use positional encoding
        self.posenc_depth = opt.posenc_depth # Number of bases for positional encoding
        if self.posenc:
            self.barf_c2f = opt.barf_c2f # use bundle adjustment in pos enc
            self.posenc_network = \
                BundleAdjustingPositionalEncoding(self.barf_c2f, self.posenc_depth, self.device) if self.barf_c2f \
                    else PositionalEncoding(self.posenc_depth, self.device)
        # network
        self.mlp = torch.nn.ModuleList()
        self.define_network()

    def define_network(self):
        input_2D_dim = 2 + 4 * self.posenc_depth if self.posenc else 2
        # point-wise RGB prediction
        L = [(input_2D_dim, 256), (256, 256), (256, 256), (256, 256), (256, 3)]
        for li, (k_in, k_out) in enumerate(L):
            linear = torch.nn.Linear(k_in, k_out)
            if self.barf_c2f and li == 0:
                # rescale first layer init (distribution was for pos.enc. but only xy is first used)
                scale = np.sqrt(input_2D_dim / 2.0)
                linear.weight.data *= scale
                linear.bias.data *= scale
            self.mlp.append(linear)

    def forward(self, coord_2D): # [B,...,3]
        if self.posenc:
            points_enc = self.posenc_network.forward(coord_2D)
            points_enc = torch.cat([coord_2D, points_enc], dim=-1) # [B,...,6L+3]
        else: 
            points_enc = coord_2D
        feat = points_enc
        # extract implicit features
        for li, layer in enumerate(self.mlp):
            # apply layer
            feat = layer(feat)
            # apply relu activation function
            if li != len(self.mlp) - 1:
                feat = F.relu(feat)
        # finally, apply sigmoid to get the RGB values, sigmoid returns values in the range [0, 1]
        rgb = feat.sigmoid_() # [B,...,3]
        return rgb

class BundleAdjustingPositionalEncoding(nn.Module):
    # Add the positional encoding from the Neural Image Function here.
    def __init__(self, barf_c2f, depth, device):
        super().__init__()
        self.barf_c2f = barf_c2f # use bundle adjustment in pos enc
        self.depth = depth # L
        self.device = device
        self.progress = torch.nn.Parameter(torch.tensor(0.)) # use Parameter so it could be checkpointed
        self.posEncNetwork = PositionalEncoding(self.depth, self.device)
    
    def forward(self, coord_2D): # [B,...,3]
        input_enc = self.posEncNetwork.forward(coord_2D)
        if self.barf_c2f is not None:
            start, end = self.barf_c2f  # e.g. start=0.2, end=0.6
            
            if self.progress.data < start:
                # Before 20% progress, set alpha to 0
                alpha = 0
            elif self.progress.data < end:
                # Between 20% and 60%, apply coarse-to-fine progression
                alpha = (self.progress.data - start) / (end - start) * self.depth
            else:
                # After 60%, use full frequency range
                alpha = self.depth
            
            # Calculate weights for each frequency band
            k = torch.arange(self.depth, dtype=torch.float32, device=self.device)
            weight = (1 - (alpha - k).clamp_(min=0,max=1).mul_(np.pi).cos_()) / 2
            
            # Apply frequency band weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1, self.depth) * weight).view(*shape)
        return input_enc

class PositionalEncoding(nn.Module):
    # Use the BundleAdjPosEnc here but in a way that the barf is essentially ignored
    # or do it the other way around - use PosEnc in barf, and apply barf c2f there
    def __init__(self, depth, device):
        super().__init__()
        self.depth = depth # L
        self.device = device
    
    def forward(self, coord_2D): # [B,...,3]
        shape = coord_2D.shape
        freq = 2 ** torch.arange(self.depth, dtype=torch.float32, device=self.device) * np.pi # [L]
        spectrum = coord_2D[...,None] * freq # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2) # [B,...,N,2,L]
        return input_enc.view(*shape[:-1], -1) # [B,...,2NL]
