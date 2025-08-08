#!/usr/bin/env python3
"""
Test script to verify your setup and dataset loading.
Run this before starting training to make sure everything works.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append('.')

try:
    from custom_dataset import CustomVideoDataset
    print("✓ Custom dataset imported successfully")
except ImportError as e:
    print(f"✗ Error importing custom dataset: {e}")
    sys.exit(1)

def test_dataset_loading():
    """Test if the dataset can load your videos"""
    print("\n=== Testing Dataset Loading ===")
    
    # Test with your data structure
    video_dir = "./data/train"
    
    if not os.path.exists(video_dir):
        print(f"✗ Video directory {video_dir} does not exist!")
        print("Please make sure your videos are in the correct location.")
        return False
    
    print(f"✓ Found video directory: {video_dir}")
    
    # List videos in the directory
    video_files = list(Path(video_dir).glob("*.mp4")) + list(Path(video_dir).glob("*.avi")) + list(Path(video_dir).glob("*.mov"))
    print(f"✓ Found {len(video_files)} video files:")
    for video_file in video_files:
        print(f"  - {video_file.name}")
    
    if len(video_files) == 0:
        print("✗ No video files found!")
        return False
    
    # Test dataset creation
    try:
        dataset = CustomVideoDataset(
            video_dir=video_dir,
            crop_size=(384, 512),
            seq_len=24,
            traj_per_sample=768,
            limit_samples=5  # Just test with a few samples
        )
        print(f"✓ Dataset created successfully with {len(dataset)} samples")
        
        # Test loading a sample
        sample, gotit = dataset[0]
        if gotit:
            print(f"✓ Sample loaded successfully")
            print(f"  - Video shape: {sample.video.shape}")
            print(f"  - Trajectory shape: {sample.trajectory.shape}")
            print(f"  - Visibility shape: {sample.visibility.shape}")
        else:
            print("✗ Failed to load sample")
            return False
            
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        return False
    
    return True

def test_dependencies():
    """Test if all required dependencies are available"""
    print("\n=== Testing Dependencies ===")
    
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'pytorch_lightning.lite',
        'cv2',
        'tqdm',
        'tensorboard'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def test_checkpoints():
    """Test if required checkpoints are available"""
    print("\n=== Testing Checkpoints ===")
    
    checkpoint_dir = "./checkpoints"
    required_checkpoints = [
        "baseline_online.pth",
        "baseline_offline.pth"
    ]
    
    if not os.path.exists(checkpoint_dir):
        print(f"✗ Checkpoint directory {checkpoint_dir} does not exist!")
        print("Please create it and download the required checkpoints.")
        return False
    
    print(f"✓ Found checkpoint directory: {checkpoint_dir}")
    
    missing_checkpoints = []
    for checkpoint in required_checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        if os.path.exists(checkpoint_path):
            print(f"✓ {checkpoint}")
        else:
            print(f"✗ {checkpoint} - MISSING")
            missing_checkpoints.append(checkpoint)
    
    if missing_checkpoints:
        print(f"\nMissing checkpoints: {', '.join(missing_checkpoints)}")
        print("Please download them with:")
        for checkpoint in missing_checkpoints:
            if "online" in checkpoint:
                print(f"wget https://huggingface.co/facebook/cotracker3/resolve/main/{checkpoint} -O {checkpoint_dir}/{checkpoint}")
            elif "offline" in checkpoint:
                print(f"wget https://huggingface.co/facebook/cotracker3/resolve/main/{checkpoint} -O {checkpoint_dir}/{checkpoint}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("CoTracker Fine-tuning Setup Test")
    print("=" * 40)
    
    all_tests_passed = True
    
    # Test dependencies
    if not test_dependencies():
        all_tests_passed = False
    
    # Test checkpoints
    if not test_checkpoints():
        all_tests_passed = False
    
    # Test dataset loading
    if not test_dataset_loading():
        all_tests_passed = False
    
    print("\n" + "=" * 40)
    if all_tests_passed:
        print("✓ All tests passed! You're ready to start fine-tuning.")
        print("\nTo start training, run:")
        print("./run_finetuning.sh")
    else:
        print("✗ Some tests failed. Please fix the issues above before starting training.")
        sys.exit(1)

if __name__ == "__main__":
    main() 