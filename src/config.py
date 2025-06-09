"""Configuration handler for the BARF model."""
import os
import yaml
from easydict import EasyDict as edict

class Config:
    """Handles loading and processing of configuration files."""
    def __init__(self, base_path):
        """Initialize configuration with base path to data directory.
        
        Args:
            base_path (str): Path to directory containing options.yaml and data
        """
        self.base_path = base_path
        self.config_path = os.path.join(base_path, 'options.yaml')
        self.opts = self._load_config()
        self._process_paths()
        self._set_defaults()

    def _load_config(self):
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return edict(config)

    def _process_paths(self):
        """Process and update paths relative to base directory."""
        # Update paths to be relative to base_path
        self.opts.img_dir = os.path.join(self.base_path, 'rgb')
        self.opts.hom_dir = os.path.join(self.base_path, 'homography')
        self.opts.output_path = os.path.join(self.base_path, 'output')

    def _set_defaults(self):
        """Set default values for required parameters."""
        # Image parameters
        self.opts.setdefault('H', 360)
        self.opts.setdefault('W', 480)
        self.opts.setdefault('dataset_images', [1, 2, 3])
        self.opts.setdefault('rescale_factor', 0.5)

        # Model behavior
        self.opts.setdefault('estimate_homs', True)
        self.opts.setdefault('debug', True)
        self.opts.setdefault('vis_hom', False)
        self.opts.setdefault('tb', False)
        self.opts.setdefault('use_sidar', False)
        self.opts.setdefault('use_cropped_images', True)

        # Fixed parameters
        self.opts.setdefault('device', 'cuda')
        self.opts.setdefault('max_iter', 2000)

        # BARF parameters
        self.opts.setdefault('posenc', True)
        self.opts.setdefault('posenc_depth', 8)
        self.opts.setdefault('barf_c2f', (0.2, 0.6))

        # Optimizer settings
        if 'optim' not in self.opts:
            self.opts.optim = edict()
        self.opts.optim.setdefault('lr', 1.e-3)
        self.opts.optim.setdefault('lr_warp', 1.e-3)
        self.opts.optim.setdefault('algo', 'Adam')

        # Homography estimation settings
        if 'homography_estimation' not in self.opts:
            self.opts.homography_estimation = edict()
        self.opts.homography_estimation.setdefault('matcher', 'lightglue')
        self.opts.homography_estimation.setdefault('feature_extractor', 'disk-depth')
        self.opts.homography_estimation.setdefault('num_features', 2048)
        self.opts.homography_estimation.setdefault('min_inliers', 50)

        # Warp settings
        if 'warp' not in self.opts:
            self.opts.warp = edict()
        self.opts.warp.setdefault('dof', 8)
        self.opts.warp.setdefault('fix_first', True)
        self.opts.warp.setdefault('type', 'homography')

    def get_opts(self):
        """Return the processed configuration options."""
        return self.opts 