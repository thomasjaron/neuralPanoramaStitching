"""This module contains the model definition."""
import os

import time
import torch
import tqdm
import PIL
import PIL.Image,PIL.ImageDraw

import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np

from config import Config
from torchvision import transforms
from easydict import EasyDict as edict
from networks import HomographyFunction
from utils import Warp, Utils
from datasets import Dataset, CustomImageLoader
import torchmetrics

# Code structure adapted from BARF
# https://github.com/chenhsuanlin/bundle-adjusting-NeRF

class ModelBase(pl.LightningModule):
    """Main Model class"""
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.it = 0
        self.vis_it = 0
        self.homography_function = None
        self.warp = None
        # create visualization and output directories
        self.vis_path = f"{self.opt.output_path}/vis"
        os.makedirs(self.vis_path, exist_ok=True)

        self.timer = None
        self.loader = None

        # Depending on set_estimated_homs, we either set the estimated homographies as
        # starting values or learn them during training
        self.estimated_homographies = None if self.opt.set_estimated_homs else -1

        self.images = None
        self.estimated_homographies_sl3 = None
        self.gt_homs_sl3 = None

        # Ground Truth image needed for visualization and comparison
        self.gt_img = None
        self.gt_img_mask = None

    def setup(self, stage):
        super().setup(stage)
        self.opt.batch_size = self.trainer.datamodule.train_dataloader().batch_size
        self.homography_function = HomographyFunction(self.opt, self.logger).to(self.opt.device)
        self.warp = Warp(self.opt)

    def _start_timer(self):
        self.timer = edict(start=time.time(), it_mean=None)
        self.loader = tqdm.trange(self.opt.max_iter, desc="Training", leave=False)

    def configure_optimizers(self):
        neural_image_lr = 0.0 if self.opt.loss_weights[2] != 0 else self.opt.optim.lr
        optim_list = [
            dict(params=self.homography_function.neural_image.parameters(), name="neural_image", lr=neural_image_lr),
            dict(params=self.homography_function.warp_param.parameters(), name="homography_function", lr=self.opt.optim.lr_warp),
        ]
        optimizer = getattr(torch.optim, self.opt.optim.algo)
        self.optim = optimizer(optim_list)
        return self.optim
    
    def initialize_dataset(self, images, gt_img=None):
        dataset = Dataset(self.opt)
        images, estimated_homographies, gt_homographies, keypoints = dataset.setup_dataset(images, gt_img)
        if gt_img is not None:
            gt_img = dataset._resize_image(gt_img)
        del dataset
        return images, estimated_homographies, gt_homographies, keypoints, gt_img

    @torch.no_grad()
    def predict_entire_image(self):
        """Retrieve the full size image from the implicit neural image function"""
        xy_grid = self.warp.get_normalized_pixel_grid(hw=(self.opt.output_H, self.opt.output_W))[:1]
        rgb = self.homography_function.neural_image.forward(xy_grid) # [B, HW, 3]
        image = rgb.view(self.opt.output_H, self.opt.output_W, 3).detach().cpu().permute(2, 0, 1)
        return image

    @torch.no_grad()
    def visualize_patches(self, opt, warp_param, pred_img=None):
        '''Visualize the homographies on a white canvas.'''
        box_colors = [
        "#ff0000",  # Red
        "#40afff",  # Sky Blue
        "#9314ff",  # Purple
        "#ffd700",  # Gold
        "#00ff00",  # Green
        "#ff4500",  # Orange Red
        "#1e90ff",  # Dodger Blue
        "#8a2be2",  # Blue Violet
        "#ffa500",  # Orange
        "#7cfc00",  # Lawn Green
        "#ff1493",  # Deep Pink
        "#00ced1",  # Dark Turquoise
        "#ba55d3",  # Medium Orchid
        "#f0e68c",  # Khaki
        "#00fa9a",  # Medium Spring Green
        "#dc143c",  # Crimson
        "#4682b4",  # Steel Blue
        "#ff69b4",  # Hot Pink
        "#32cd32",  # Lime Green
        "#800000"   # Maroon
    ]
        box_colors = list(map(self.warp.colorcode_to_number, box_colors))
        self.box_colors = np.array(box_colors).astype(int)
        # Prepare Canvas
        if pred_img is not None:
            Utils.print_image(pred_img, self.vis_path, f'{self.vis_it}-pred', multiplier=255)
            pred_img = pred_img.permute(1, 2, 0).numpy()
            image_pil = transforms.ToPILImage()(pred_img).convert("RGBA")
        else:
            image_pil = PIL.Image.new("RGBA", (self.opt.output_W, self.opt.output_H), color=(255, 255, 255, 255))

        # prepare canvas for homography visualization
        draw_pil = PIL.Image.new("RGBA",image_pil.size,(0,0,0,0))
        draw = PIL.ImageDraw.Draw(draw_pil)
        corners_all = self.warp.warp_corners(warp_param)
        corners_all = self.warp.normalize_coordinates(corners_all)
        # prepare canvas for homography mask
        bw_image_pil = PIL.Image.new("RGBA", (self.opt.output_W, self.opt.output_H), color=(0, 0, 0, 255))
        bw_draw_pil = PIL.Image.new("RGBA", bw_image_pil.size, (0,0,0,0))
        bw_draw = PIL.ImageDraw.Draw(bw_draw_pil)

        # Visualize Homographies as polygons
        for i,corners in enumerate(corners_all):
            P = [tuple(float(n) for n in corners[j]) for j in range(4)]
            draw.line([P[0],P[1],P[2],P[3],P[0]],fill=tuple(self.box_colors[i]),width=3)
            bw_draw.polygon(P, fill=(255, 255, 255, 255))
        # overlay homographies on image
        image_pil.alpha_composite(draw_pil)
        image_tensor = transforms.functional.to_tensor(image_pil.convert("RGB"))
        Utils.print_image(image_tensor, self.vis_path, f'{self.vis_it}-homographies', multiplier=255)
        # Draw Homography Mask
        bw_image_pil.alpha_composite(bw_draw_pil)
        bw_image_tensor = transforms.functional.to_tensor(bw_image_pil.convert("RGB"))
        bw_image_tensor = (bw_image_tensor > 0).float()

        # Draw keypoint matches as lines, transformed by homographies
        if self.kps is not None:
            homs = self.warp.sl3_to_SL3_stack(warp_param)
            for i in range(self.opt.batch_size):
                for j in range(self.opt.batch_size):
                    if i == j:
                        continue
                    if self.kps[i][j].shape[0] < self.opt.homography_estimation.min_inliers:
                        continue
                    hom_i = homs[i]
                    hom_j = homs[j]
                    kps_i, kps_j = self.warp.project_keypoint_match(self.kps[i][j][:, 0, :], self.kps[i][j][:, 1, :], hom_i, hom_j)
                    kps_i = kps_i.detach().cpu().numpy()
                    kps_j = kps_j.detach().cpu().numpy()
                    for pi, pj in zip(kps_i, kps_j):
                        draw.line([(pi[0], pi[1]), (pj[0], pj[1])], fill=tuple(self.box_colors[i]), width=1)
        image_pil.alpha_composite(draw_pil)
        image_tensor = transforms.functional.to_tensor(image_pil.convert("RGB"))
        Utils.print_image(image_tensor, self.vis_path, f'{self.vis_it}-homographies-and-kps', multiplier=255)
            
        # Apply GT Mask on GT Image (only available if SIDAR is used, only computed once)
        if self.gt_img_mask is None and self.gt_img is not None:
            gt_image_pil = PIL.Image.new("RGBA", (self.opt.output_W, self.opt.output_H), color=(0, 0, 0, 255))
            gt_draw_pil = PIL.Image.new("RGBA", gt_image_pil.size, (0,0,0,0))
            gt_draw = PIL.ImageDraw.Draw(gt_draw_pil)
            corners_gt = self.warp.warp_corners(self.gt_homs_sl3)
            corners_gt = self.warp.normalize_coordinates(corners_gt)
            for corners in corners_gt:
                P = [tuple(float(n) for n in corners[j]) for j in range(4)]
                gt_draw.polygon(P, fill=(255, 255, 255, 255))
            gt_image_pil.alpha_composite(gt_draw_pil)
            self.gt_img_mask = transforms.functional.to_tensor(gt_image_pil.convert("RGB"))
            Utils.print_image(self.gt_img_mask, self.vis_path, f'gt-img-mask', multiplier=255)
            self.gt_img_mask = (self.gt_img_mask > 0).float().to(self.device)
            masked_gt = self.gt_img.to(self.device) * self.gt_img_mask
            Utils.print_image(masked_gt, self.vis_path, f'gt-img-masked', multiplier=255)
            self.masked_gt_img = masked_gt.to(self.device)

        Utils.print_image(bw_image_tensor, self.vis_path, f'{self.vis_it}-mask', multiplier=255)
        masked_pred = transforms.functional.to_tensor(pred_img) * bw_image_tensor
        Utils.print_image(masked_pred, self.vis_path, f'{self.vis_it}-masked-pred', multiplier=255)
        return masked_pred.to(self.device)
        
    def training_step(self, images, batch_idx):
        """Use the nn.Modules here and let them interact
        Basically, loss calculation should happen here."""

        # initializations for the first iteration
        if not self.timer:
            self.initialize_model(images)

        # If blending is active, learn the neural image only after a certain amount of iterations.
        if self.opt.loss_weights[2] == 1:  # Check if blending loss weight is 1
            if self.it / self.opt.max_iter >= self.opt.loss_threshold:
                for group in self.optim.param_groups:
                    if group.get("name") == "neural_image" and group["lr"] == 0.0:
                        group["lr"] = self.opt.optim.lr  # Set it to the actual learning rate

        self.timer.it_start = time.time()
        # Forward pass 
        _ = self.homography_function.forward(self.it)
        loss = self.homography_function.compute_loss(self.images, self.kps, self.it)

        # Reset homography estimation for the fixed image
        if self.opt.warp.fix_first:
            if self.opt.use_sidar:
                self.homography_function.warp_param.weight.data[0] = self.gt_homs_sl3[0]
            else:
                self.homography_function.warp_param.weight.data[0] = 0

        # Visualize progress periodically
        if (self.it) % self.opt.log_image == 0 or self.it == self.opt.max_iter - 1:
            frame = self.predict_entire_image()
            if self.opt.vis_hom:
                masked_pred = self.visualize_patches(self.opt, self.homography_function.warp_param.weight, pred_img=frame)
                # SSIM 
                if self.opt.use_sidar:
                    self.logger.experiment.add_scalar("Metrics/SSIM", self.ssim_loss(masked_pred, self.masked_gt_img), self.it)
            self.vis_it += 1

        # log homography error if groundtruth homographies are provided
        if self.opt.tb and self.it % self.opt.log_scalar == 0 or self.it == self.opt.max_iter - 1:
           self.log_metrics(loss)

        # Update c2f progress
        if self.opt.posenc and self.opt.barf_c2f:
            self.homography_function.neural_image.posenc_network.progress.data.fill_(self.it / self.opt.max_iter)

        # Finish iteration
        self.it += 1
        self.loader.update(1)  # Update tqdm progress bar    
        return loss.render
    
    def initialize_model(self, images):
        if self.opt.use_sidar:
            self.gt_img = images[0]
            images = images[1:]
        self.images, self.estimated_homographies, self.gt_homs, self.kps, self.gt_img = self.initialize_dataset(images, self.gt_img)
        if self.opt.use_sidar:
            self.opt.output_H = int(self.opt.H * self.opt.rescale_factor)
            self.opt.output_W = int(self.opt.W * self.opt.rescale_factor)
        # Update batch size based on filtered dataset
        self.estimated_homographies_sl3 = self.warp.SL3_homs_to_sl3_stack(self.estimated_homographies[0])
        self.opt.batch_size = len(self.images)
        
        # Reinitialize networks with new batch size
        self.homography_function = HomographyFunction(self.opt, self.logger).to(self.opt.device)
        self.warp = Warp(self.opt)
        
        if self.opt.use_sidar:
            self.gt_homs_sl3 = self.warp.SL3_homs_to_sl3_stack(self.gt_homs)
        
        # Reconfigure optimizer with new network parameters
        if self.opt.set_estimated_homs:
            self.homography_function.initialize_warp_params(self.estimated_homographies)
        optim = self.optimizers()
        for group in optim.param_groups:
            if group.get('name') == 'homography_function':
                group['params'] = list(self.homography_function.warp_param.parameters())
            elif group.get('name') == 'neural_image':
                group['params'] = list(self.homography_function.neural_image.parameters())
        self._start_timer()
    
    def log_metrics(self, loss):
        # log total loss (with weighting) and single parts of the loss (without weighting)
        self.logger.experiment.add_scalar("Loss/Render", loss.render, self.it)
        self.logger.experiment.add_scalar("Loss/RGB", loss.rgb, self.it)
        self.logger.experiment.add_scalar("Loss/Keypoints", loss.kps, self.it)
        self.logger.experiment.add_scalar("Loss/Gradients", loss.grad, self.it)
        # log homography errors between prediction and SIDAR / prediction and cv2.findHomography results
        if self.opt.use_sidar:
            self.logger.experiment.add_scalar("Metrics/EST_TO_SIDAR", self.homography_error(self.homography_function.warp_param.weight, self.gt_homs_sl3), self.it)
        if self.opt.use_sidar and self.estimated_homographies is not None:
            self.logger.experiment.add_scalar("Loss/RANSAC_TO_SIDAR", self.homography_error(self.estimated_homographies_sl3, self.gt_homs_sl3), self.it)

    def ssim_loss(self, masked_pred, masked_gt_img):
        ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        return ssim(masked_pred.unsqueeze(0), masked_gt_img.unsqueeze(0))
    
    def homography_error(self, pred_hom, gt_hom):
        return (pred_hom-gt_hom).norm(dim=-1).mean() #torch.mean((pred_hom - gt_hom)**2)

if __name__ == "__main__":

    # Load configuration from data directory
    config = Config('./data_example')
    opts = config.get_opts()

    dataloader = CustomImageLoader(opts)

    logger = TensorBoardLogger(save_dir=opts.output_path, name="tb_logs") if opts.tb else False

    # train model
    trainer = pl.Trainer(
        max_steps=opts.max_iter,
        enable_progress_bar=False,
        accelerator="gpu" if opts.device == "cuda" else "cpu",
        devices="1",
        logger=logger,
        enable_checkpointing=False,
        )
    trainer.fit(model=ModelBase(opts), datamodule=dataloader)
