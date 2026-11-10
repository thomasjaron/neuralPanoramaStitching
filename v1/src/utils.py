import torch
import scipy
import imageio
from torchvision import transforms

# Code structure adapted from BARF
# https://github.com/chenhsuanlin/bundle-adjusting-NeRF

class Utils:
    """Utility functions for various purposes"""
    @staticmethod
    def move_to_device(x, device):
        """Recursively move the given input value to the selected device (cpu or cuda)."""
        if isinstance(x, dict):
            for k, v in x.items():
                x[k] = Utils.move_to_device(v, device)
        elif isinstance(x, list):
            for i, e in enumerate(x):
                x[i] = Utils.move_to_device(e, device)
        elif isinstance(x, tuple) and hasattr(x, "_fields"):  # collections.namedtuple
            dd = x._asdict()
            dd = Utils.move_to_device(dd, device)
            return type(x)(**dd)
        elif isinstance(x, torch.Tensor):
            return x.to(device=device)
        return x
    
    @staticmethod
    def print_image(img, path, name, multiplier=1):
        """Takes an image tensor and saves it into a .png file."""
        if img.is_cuda:
            img = img.cpu()
        if len(img.shape) < 3:
            print("Image has less than 3 dimensions. Cannot save.")
            return
        if len(img.shape) == 4:
            img = img[0]
        imageio.imsave(f"{path}/{name}.png", (img * multiplier).byte().permute(1, 2, 0).numpy())

class Warp:
    """Functions for linalg operations and implicit functions"""
    def __init__(self, opt):
        self.opt = opt
        self.max_h = opt.H
        self.crop_h = self.max_h * opt.rescale_factor
        self.max_w = opt.W
        self.crop_w = self.max_w * opt.rescale_factor
        self.norm_h = self.max_h / max(self.max_h, self.max_w)
        self.norm_w = self.max_w / max(self.max_h, self.max_w)
        self.output_h = opt.output_H
        self.output_w = opt.output_W
        self.batch_size = opt.batch_size
        self.device = opt.device
        self.warp_type = opt.warp.type
        self.dof = opt.warp.dof

    def warp_corners(self, warp_param):
        y_crop = (self.max_h//2-self.crop_h//2,self.max_h//2+self.crop_h//2)
        x_crop = (self.max_w//2-self.crop_w//2,self.max_w//2+self.crop_w//2)
        Y = [((y+0.5)/self.opt.H*2-1)*(self.opt.H/max(self.opt.H,self.opt.W)) for y in y_crop]
        X = [((x+0.5)/self.opt.W*2-1)*(self.opt.W/max(self.opt.H,self.opt.W)) for x in x_crop]
        corners = [(X[0],Y[0]),(X[0],Y[1]),(X[1],Y[1]),(X[1],Y[0])]
        corners = torch.tensor(corners,dtype=torch.float32,device=self.device).repeat(self.batch_size,1,1)
        corners_warped = self.warp_grid(corners,warp_param)
        return corners_warped

    def colorcode_to_number(self, code):
        ords = [ord(c) for c in code[1:]]
        ords = [n-48 if n<58 else n-87 for n in ords]
        rgb = (ords[0]*16+ords[1],ords[2]*16+ords[3],ords[4]*16+ords[5])
        return rgb

    def to_hom(self, matrix):
        """Convert a matrix to the homogenous format."""
        # get homogeneous coordinates of the input
        mat_hom = torch.cat([matrix, torch.ones_like(matrix[..., :1])], dim=-1)
        return mat_hom

    def sl3_to_SL3(self, h):
        """homography: directly expand matrix exponential"""
        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.6151&rep=rep1&type=pdf
        h1, h2, h3, h4, h5, h6, h7, h8 = h.chunk(8, dim=-1)
        A = torch.stack([torch.cat([h5, h3, h1], dim=-1),
                         torch.cat([h4, -h5-h6, h2], dim=-1),
                         torch.cat([h7, h8, h6], dim=-1)], dim=-2)
        # print(h.chunk(8, dim=-1))
        H = A.matrix_exp()
        return H
    
    def matrix_log(self, H):
        """Compute the matrix logarithm using SciPy."""
        H_np = H.detach().cpu().numpy()
        A_np = scipy.linalg.logm(H_np)
        A = torch.tensor(A_np, dtype=H.dtype, device=H.device)
        return A
    
    def SL3_to_sl3(self, H):
        """Inverse of sl3_to_SL3: Compute the parameters h from the SL(3) matrix."""
        # Convert the PyTorch tensor to a NumPy array

        # Check if matrix determinant is 1.

        A = self.matrix_log(H)
        
        # Extract elements of A
        h1 = A[0, 2].unsqueeze(-1)
        h2 = A[1, 2].unsqueeze(-1)
        h3 = A[0, 1].unsqueeze(-1)
        h4 = A[1, 0].unsqueeze(-1)
        h5 = A[0, 0].unsqueeze(-1)
        h6 = A[2, 2].unsqueeze(-1)
        h7 = A[2, 0].unsqueeze(-1)
        h8 = A[2, 1].unsqueeze(-1)

        # print(A)
    
        # Combine elements back into h
        h = torch.cat([h1, h2, h3, h4, h5, h6, h7, h8], dim=-1)
        return h

    def SL3_homs_to_sl3_stack(self, homs):
        flattened_homs = []
        for H in homs:
            h = self.SL3_to_sl3(H)
            flattened_homs.append(h)
        return torch.stack(flattened_homs)
    
    def sl3_to_SL3_stack(self, sl3s):
        new_homs = []
        for h in sl3s:
            H = self.sl3_to_SL3(h)
            new_homs.append(H)
        return torch.stack(new_homs)

    def get_normalized_pixel_grid(self, hw=None, crop=False):
        """Create a pixel grid fitting to the output image width and height,
        which optionally can be cropped to fit the input image width and height"""
        # prepare grid dimensions
        if hw or crop:
            if hw:
                h = hw[0]
                w = hw[1]
                self.y_crop = (
                    self.max_h // 2 - h // 2, self.max_h // 2 + h // 2
                    )
                self.x_crop = (
                    self.max_w // 2 - w // 2, self.max_w // 2 + w // 2
                    )
            elif crop:
                self.y_crop = (
                    self.max_h // 2 - self.crop_h // 2, self.max_h // 2 + self.crop_h // 2
                    )
                self.x_crop = (
                    self.max_w // 2 - self.crop_w // 2, self.max_w // 2 + self.crop_w // 2
                    )
            y_range = (
                (torch.arange(
                    *(self.y_crop),
                    dtype=torch.float32,
                    device=self.device) + 0.5
                ) / self.max_h * 2 - 1) * self.norm_h
            x_range = (
                (torch.arange(
                    *(self.x_crop),
                    dtype=torch.float32,
                    device=self.device) + 0.5
                ) / self.max_w * 2 - 1) * self.norm_w
            Y, X = torch.meshgrid(y_range, x_range)  # [H, W]
            xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW, 2]
            xy_grid = xy_grid.repeat(self.batch_size, 1, 1)  # [B, HW, 2]
            return xy_grid
        else:
            y_range = ((torch.arange(
                    self.max_h,
                    dtype=torch.float32,
                    device=self.device) + 0.5
                ) / self.max_h * 2 - 1) * self.norm_h
            x_range = ((torch.arange(
                    self.max_w,
                    dtype=torch.float32,
                    device=self.device) + 0.5
                ) / self.max_w * 2 - 1) * self.norm_w
            Y, X = torch.meshgrid(y_range, x_range)  # [H, W]
            xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW, 2]
            xy_grid = xy_grid.repeat(self.batch_size, 1, 1)  # [B, HW, 2]
            return xy_grid

    def warp_grid(self, xy_grid, warp):
        """Perform a homography warp onto the input grid / image.
        warp is a matrix containing sl3 representations of homographies."""
        xy_grid_hom = self.to_hom(xy_grid)
        warp_matrix = self.sl3_to_SL3(warp)
        warped_grid_hom = xy_grid_hom @ warp_matrix.transpose(-2, -1)
        warped_grid = warped_grid_hom[..., :2] / (warped_grid_hom[..., 2:] + 1e-8)  # [B,HW,2]
        return warped_grid

    def normalize_coordinates(self, coords, is_keypoints=False):
        """Normalize coordinates from homogeneous space to image space.
        
        Args:
            coords (torch.Tensor): Coordinates to normalize. Shape [..., 2] or [..., 3] if homogeneous
            is_keypoints (bool): Whether the input coordinates are keypoints (needs homogeneous division)
        
        Returns:
            torch.Tensor: Normalized coordinates in image space
        """
        if is_keypoints:
            # Divide by homogeneous coordinate
            coords = coords / coords[..., 2:3]
            coords = coords[..., :2]  # Remove homogeneous dimension
            
        # Scale based on aspect ratio
        max_dim = max(self.opt.H, self.opt.W)
        coords[..., 0] = coords[..., 0] * max_dim / self.opt.W
        coords[..., 1] = coords[..., 1] * max_dim / self.opt.H
        
        # Convert from [-1, 1] to image space
        coords = (coords + 1) / 2  # Convert to [0, 1] range
        coords[..., 0] = coords[..., 0] * self.opt.W
        coords[..., 1] = coords[..., 1] * self.opt.H
        
        # Center in output image
        coords[..., 0] += (self.opt.output_W - self.opt.W) / 2
        coords[..., 1] += (self.opt.output_H - self.opt.H) / 2
        
        return coords

    def project_keypoint_match(self, kps_i, kps_j, hom_i, hom_j):
        """Project the keypoint pairs to the reference image.
        
        Args:
            kps_i (torch.Tensor): Keypoints from first image [N, 2]
            kps_j (torch.Tensor): Keypoints from second image [N, 2]
            hom_i (torch.Tensor): Homography matrix for first image [3, 3]
            hom_j (torch.Tensor): Homography matrix for second image [3, 3]
        
        Returns:
            tuple: Normalized keypoint coordinates for both images
        """
        # Convert to homogeneous coordinates
        hom_ones = torch.ones((kps_i.shape[0], 1), device=self.device)
        kps_i_hom = torch.cat((kps_i, hom_ones), dim=1)  # [N, 3]
        kps_j_hom = torch.cat((kps_j, hom_ones), dim=1)  # [N, 3]
        
        # Apply homography transformations
        kps_i_transformed = kps_i_hom @ hom_i.T  # [N, 3]
        kps_j_transformed = kps_j_hom @ hom_j.T  # [N, 3]
        
        # Normalize coordinates to image space
        kps_i_norm = self.normalize_coordinates(kps_i_transformed, is_keypoints=True)
        kps_j_norm = self.normalize_coordinates(kps_j_transformed, is_keypoints=True)
        
        return kps_i_norm, kps_j_norm