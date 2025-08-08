import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_video
from cotracker.datasets.utils import CoTrackerData


class CustomVideoDataset(Dataset):
    """
    Custom dataset class for fine-tuning CoTracker on your personal videos.
    
    This class loads videos from a directory and prepares them for training.
    The videos will be used to generate pseudo-labels during training.
    """
    
    def __init__(
        self,
        video_dir,
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        random_frame_rate=False,
        random_seq_len=False,
        random_resize=False,
        limit_samples=None,
        video_extensions=('.mp4', '.avi', '.mov', '.mkv')
    ):
        """
        Args:
            video_dir: Path to directory containing your videos
            crop_size: Size to crop videos to (height, width)
            seq_len: Number of frames to use per sequence
            traj_per_sample: Number of trajectories to sample (will be filled with pseudo-labels)
            random_frame_rate: Whether to use random frame rates during training
            random_seq_len: Whether to use random sequence lengths during training
            random_resize: Whether to randomly resize videos
            limit_samples: Maximum number of videos to use (None for all)
            video_extensions: File extensions to consider as videos
        """
        super().__init__()
        
        self.video_dir = Path(video_dir)
        self.crop_size = crop_size
        self.seq_len = seq_len
        self.traj_per_sample = traj_per_sample
        self.random_frame_rate = random_frame_rate
        self.random_resize = random_resize
        self.random_seq_len = random_seq_len
        
        # Find all video files
        self.video_files = []
        for ext in video_extensions:
            self.video_files.extend(self.video_dir.glob(f"*{ext}"))
            self.video_files.extend(self.video_dir.glob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        self.video_files = sorted(list(set(self.video_files)))
        
        # Limit number of samples if specified
        if limit_samples is not None:
            self.video_files = self.video_files[:limit_samples]
        
        print(f"Found {len(self.video_files)} videos in {video_dir}")
        
        # Filter out videos that are too short or too large
        self.valid_videos = []
        for video_path in self.video_files:
            try:
                # Quick check to see if video has enough frames and isn't too large
                file_size = video_path.stat().st_size
                if file_size > 100 * 1024 * 1024:  # Skip videos larger than 100MB
                    print(f"Skipping large video: {video_path.name} ({file_size / 1024 / 1024:.1f}MB)")
                    continue
                    
                rgbs, _, _ = read_video(str(video_path), output_format="TCHW", pts_unit="sec")
                if len(rgbs) >= seq_len:
                    self.valid_videos.append(video_path)
                else:
                    print(f"Skipping short video: {video_path.name} ({len(rgbs)} frames)")
            except Exception as e:
                print(f"Warning: Could not read {video_path}: {e}")
        
        self.video_files = self.valid_videos
        print(f"Valid videos for training: {len(self.video_files)}")
        
        # Pre-compute video lengths to avoid repeated loading
        self.video_lengths = {}
        for video_path in self.video_files:
            try:
                rgbs, _, _ = read_video(str(video_path), output_format="TCHW", pts_unit="sec")
                self.video_lengths[video_path] = len(rgbs)
            except Exception as e:
                print(f"Warning: Could not get length for {video_path}: {e}")
                self.video_lengths[video_path] = 0
    
    def crop(self, rgbs):
        """Crop videos to the specified size"""
        S = len(rgbs)
        H, W = rgbs.shape[2:]
        
        # Simple random crop
        y0 = (
            0
            if self.crop_size[0] >= H
            else np.random.randint(0, H - self.crop_size[0])
        )
        x0 = (
            0
            if self.crop_size[1] >= W
            else np.random.randint(0, W - self.crop_size[1])
        )
        
        rgbs = [
            rgb[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            for rgb in rgbs
        ]
        
        return torch.stack(rgbs)
    
    def __getitem__(self, index):
        """Get a video sample"""
        gotit = False
        
        try:
            sample, gotit = self._getitem_helper(index)
        except Exception as e:
            print(f"Error loading video {index}: {e}")
            gotit = False
        
        if not gotit:
            # Return a dummy sample if loading failed
            sample = CoTrackerData(
                video=torch.zeros(
                    (self.seq_len, 3, self.crop_size[0], self.crop_size[1])
                ),
                trajectory=torch.ones(1, 1, 1, 2),
                visibility=torch.ones(1, 1, 1),
                valid=torch.ones(1, 1, 1),
            )
        
        return sample, gotit
    
    def _getitem_helper(self, index):
        """Helper function to load a video sample"""
        video_path = self.video_files[index]
        
        try:
            # Load video with error handling
            rgbs, _, _ = read_video(str(video_path), output_format="TCHW", pts_unit="sec")
        except Exception as e:
            print(f"Failed to load video {video_path}: {e}")
            return None, False
        
        if rgbs.numel() == 0:
            return None, False
        
        seq_name = str(video_path)
        frame_rate = 1
        
        # Determine sequence length
        if self.random_seq_len:
            seq_len = np.random.randint(int(self.seq_len / 2), self.seq_len)
        else:
            seq_len = self.seq_len
        
        # Handle videos that are too short by repeating frames
        while len(rgbs) < seq_len:
            rgbs = torch.cat([rgbs, rgbs.flip(0)])
        
        if seq_len < 8:
            return None, False
        
        # Random frame rate sampling
        if self.random_frame_rate:
            max_frame_rate = min(4, int((len(rgbs) / seq_len)))
            if max_frame_rate > 1:
                frame_rate = np.random.randint(1, max_frame_rate)
        
        # Select frame range
        if seq_len * frame_rate < len(rgbs):
            start_ind = np.random.choice(len(rgbs) - (seq_len * frame_rate), 1)[0]
        else:
            start_ind = 0
        
        rgbs = rgbs[start_ind : start_ind + seq_len * frame_rate : frame_rate]
        
        assert seq_len <= len(rgbs)
        
        # Random resize if enabled
        if self.random_resize and np.random.rand() < 0.5:
            import cv2
            video = []
            rgbs = rgbs.permute(0, 2, 3, 1).numpy()
            
            for i in range(len(rgbs)):
                rgb = cv2.resize(
                    rgbs[i],
                    (self.crop_size[1], self.crop_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                video.append(rgb)
            video = torch.tensor(np.stack(video)).permute(0, 3, 1, 2)
        else:
            video = self.crop(rgbs)
        
        # Create sample with dummy trajectories (will be filled with pseudo-labels during training)
        sample = CoTrackerData(
            video=video,
            trajectory=torch.ones(seq_len, self.traj_per_sample, 2),
            visibility=torch.ones(seq_len, self.traj_per_sample),
            valid=torch.ones(seq_len, self.traj_per_sample),
            seq_name=seq_name,
        )
        
        return sample, True
    
    def __len__(self):
        return len(self.video_files)


# Example usage and training script
def create_training_script():
    """Example of how to use the custom dataset for training"""
    
    script_content = '''#!/bin/bash

# Example training script for fine-tuning CoTracker on your personal dataset

# Set your paths
DATASET_ROOT="/path/to/your/videos"
CHECKPOINT_PATH="./checkpoints/my_finetuned_model"
RESTORE_CKPT="./checkpoints/baseline_online.pth"  # or baseline_offline.pth

# Create checkpoint directory
mkdir -p $CHECKPOINT_PATH

# Training command for online model
python train_on_real_data.py \\
    --batch_size 1 \\
    --num_steps 15000 \\
    --ckpt_path $CHECKPOINT_PATH \\
    --model_name cotracker_three \\
    --save_freq 200 \\
    --sequence_len 64 \\
    --eval_datasets tapvid_stacking tapvid_davis_first \\
    --traj_per_sample 384 \\
    --save_every_n_epoch 15 \\
    --evaluate_every_n_epoch 15 \\
    --model_stride 4 \\
    --dataset_root $DATASET_ROOT \\
    --num_nodes 1 \\
    --real_data_splits 0 \\
    --num_virtual_tracks 64 \\
    --mixed_precision \\
    --random_frame_rate \\
    --restore_ckpt $RESTORE_CKPT \\
    --lr 0.00005 \\
    --real_data_filter_sift \\
    --validate_at_start \\
    --sliding_window_len 16 \\
    --limit_samples 15000

# For offline model, add --offline_model flag and use baseline_offline.pth
'''
    
    with open('train_custom_dataset.sh', 'w') as f:
        f.write(script_content)
    
    print("Created training script: train_custom_dataset.sh")


if __name__ == "__main__":
    # Example of how to use the dataset
    dataset = CustomVideoDataset(
        video_dir="./your_videos",
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        limit_samples=100
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample, gotit = dataset[0]
    if gotit:
        print(f"Sample video shape: {sample.video.shape}")
        print(f"Sample trajectory shape: {sample.trajectory.shape}")
    
    # Create training script
    create_training_script() 