# CoTracker Fine-Tuning Deep Dive

## Overview
This document provides a comprehensive, step-by-step guide to fine-tuning the CoTracker model on a custom video dataset. It covers the technical pipeline, encountered challenges, solutions, and best practices, based on a real-world session.

---

## 1. Project Structure and Data Preparation

### **Directory Layout**
```
co-tracker/
├── checkpoints/
│   └── my_finetuned_model/
├── data/
│   ├── train/
│   │   └── output_cropped1.mp4
│   └── test/
│       └── output_cropped2.mp4
...
```
- **train/**: Contains your training videos.
- **test/**: Contains your test videos for evaluation.

### **Data Requirements**
- Videos should be in a supported format (e.g., `.mp4`).
- For best results, videos should be at least 8 frames long and have a resolution divisible by the model stride (usually 4).

---

## 2. Environment Setup

### **Conda Environment**
- Use Python 3.10+ (required for some dependencies like Haiku).
- Install dependencies:
  ```bash
  conda create -n cotracker310 python=3.10
  conda activate cotracker310
  pip install torch torchvision pytorch_lightning opencv-python imageio av tqdm tensorboard
  ```
- Install any additional dependencies required by CoTracker and TAPIR.

### **Checkpoints**
- Download the baseline checkpoints:
  ```bash
  mkdir -p checkpoints
  wget https://huggingface.co/facebook/cotracker3/resolve/main/baseline_online.pth -O checkpoints/baseline_online.pth
  wget https://huggingface.co/facebook/cotracker3/resolve/main/baseline_offline.pth -O checkpoints/baseline_offline.pth
  ```

---

## 3. Custom Dataset Integration

### **Custom Dataset Class**
- Implemented `CustomVideoDataset` to load videos from a directory, crop, and preprocess them for training.
- Ensured compatibility with CoTracker's expected data format (`CoTrackerData`).
- Added logic to filter out videos that are too short or have incompatible resolutions.

---

## 4. Fine-Tuning Pipeline

### **Training Script**
- Created `train_custom_dataset.py` to:
  - Load the custom dataset.
  - Use pseudo-labeling with teacher models (including TAPIR and pre-trained CoTracker models).
  - Train the student model on pseudo-labeled data.
  - Save checkpoints and log training progress.

### **Key Training Parameters**
- `--video_dir`: Path to your training videos.
- `--restore_ckpt`: Path to the pre-trained checkpoint to start from.
- `--ckpt_path`: Where to save your fine-tuned model.
- `--sequence_len` and `--sliding_window_len`: Control the temporal window for training (should be compatible).
- `--traj_per_sample`: Number of points to track per video.
- `--batch_size`: Set to 1 for memory efficiency.
- `--mixed_precision`: Use for faster and more memory-efficient training.

### **Example Command**
```bash
python train_custom_dataset.py \
    --video_dir ./data/train \
    --ckpt_path ./checkpoints/my_finetuned_model \
    --restore_ckpt ./checkpoints/baseline_online.pth \
    --batch_size 1 \
    --num_steps 15000 \
    --sequence_len 16 \
    --sliding_window_len 16 \
    --traj_per_sample 384 \
    --mixed_precision \
    --real_data_filter_sift
```

---

## 5. Troubleshooting & Lessons Learned

### **A. Dependency Issues**
- Some libraries (e.g., Haiku) require Python 3.10+.
- PyTorch Lightning 1.6.0 may require an older pip version; using a newer Lightning version is often easier.
- PyAV is required for video loading with torchvision/imageio.

### **B. Evaluation Dataset Errors**
- The default training script tries to evaluate on datasets you may not have (e.g., `tapvid_kinetics`).
- **Solution:** Patch the script to only evaluate on datasets you actually have (e.g., `tapvid_stacking`, `tapvid_davis_first`).

### **C. Sequence Length and Window Size**
- Mismatched `sequence_len` and `sliding_window_len` can cause empty slices or shape mismatches.
- **Best Practice:** Set them to the same value, or ensure `sequence_len >= sliding_window_len`.

### **D. Out-of-Memory (OOM) Issues**
- High-resolution videos and long sequences can easily exceed GPU memory.
- **Solutions:**
  - Downscale videos before training/inference (e.g., to 512x512 or 384x384).
  - Reduce the number of frames per video.
  - Use fewer queries/points per video.
  - Use mixed precision.

### **E. Model API Differences**
- The online model expects queries of shape `[B, N, 3]` (not a full video tensor).
- For grid tracking, use the `grid_size` argument or manually create a small set of queries.
- Do not use `torch.ones_like(video)` for queries; this will create an enormous number of points and cause OOM.

### **F. Query Construction**
- For custom tracking, create a `[1, N, 3]` tensor where each row is `[t, y, x]` (frame, row, col).
- For a grid of points on the first frame:
  ```python
  grid_size = 10
  ys = torch.linspace(0, H-1, grid_size)
  xs = torch.linspace(0, W-1, grid_size)
  yy, xx = torch.meshgrid(ys, xs, indexing='ij')
  yy = yy.flatten()
  xx = xx.flatten()
  t = torch.zeros_like(yy)
  queries = torch.stack([t, yy, xx], dim=1).unsqueeze(0).to(device)  # [1, 100, 3]
  ```

---

## 6. Inference & Visualization

### **Testing the Fine-Tuned Model**
- Created `test_finetuned_model.py` to:
  - Load the fine-tuned checkpoint.
  - Load and (optionally) downscale the test video.
  - Generate a small grid of queries on the first frame.
  - Run the model and visualize the predicted tracks.

### **Example Inference Block**
```python
with torch.no_grad():
    pred_tracks, pred_visibility = model(video, queries=queries, iters=100)
```

### **Visualization**
- Used the provided `Visualizer` to save output videos with tracked points.
- Output is saved to a specified directory for easy review.

---

## 7. Best Practices & Recommendations

- **Always downscale high-res videos for training/inference.**
- **Use a small number of queries for memory efficiency.**
- **Set `sequence_len` and `sliding_window_len` to compatible values.**
- **Patch evaluation code to only use datasets you have.**
- **Test your pipeline with a single video/sample before scaling up.**
- **Monitor GPU memory usage and adjust parameters as needed.**

---

## 8. Example: End-to-End Pipeline

1. **Prepare your videos** in `data/train` and `data/test`.
2. **Downscale videos** if needed.
3. **Run the test script** to verify setup:
   ```bash
   python test_setup.py
   ```
4. **Fine-tune the model**:
   ```bash
   ./run_finetuning.sh
   ```
5. **Test the fine-tuned model**:
   ```bash
   python test_finetuned_model.py
   ```
6. **Review the output** in `test_outputs/`.

---

## 9. Lessons for Future Users

- **Start small:** Use short, low-res videos and few queries to debug the pipeline.
- **Incrementally scale up** once everything works.
- **Document your changes** to the codebase for reproducibility.
- **Keep an eye on library versions**—compatibility issues are common in research code.
- **Understand the model API**—read the docstrings and check expected input shapes.

---

## 10. References
- [CoTracker GitHub](https://github.com/facebookresearch/co-tracker)
- [TAPIR (TapNet) GitHub](https://github.com/deepmind/tapnet)
- [PyTorch Lightning Docs](https://pytorch-lightning.readthedocs.io/)
- [PyAV Docs](https://pyav.org/docs/stable/)

---

**This guide is based on a real, iterative fine-tuning session and is intended to help future users avoid common pitfalls and get the most out of CoTracker for their own data.**

## Handling Multiple Videos

### Memory Optimization
When training with multiple videos, memory management is crucial:

#### 1. DataLoader Optimizations
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=2,  # Reduced batch size
    num_workers=2,  # Limited workers
    pin_memory=False,  # Disable pin_memory
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=2,  # Reduce prefetch
)
```

#### 2. Video Size Limits
The dataset automatically filters videos:
- Skips videos larger than 100MB
- Truncates sequences longer than 30 frames during testing
- Downscales large videos automatically

#### 3. Memory Management
```python
# Clear memory after each batch
del batch, output, loss
torch.cuda.empty_cache()

# Error handling with memory cleanup
try:
    # Process batch
    pass
except Exception as e:
    torch.cuda.empty_cache()
    continue
```

### Recommended Settings for Multiple Videos

#### For 5-10 videos:
```bash
python train_custom_dataset.py \
    --batch_size 2 \
    --num_workers 2 \
    --sequence_len 16 \
    --limit_samples 5 \
    --mixed_precision
```

#### For 10+ videos:
```bash
python train_custom_dataset.py \
    --batch_size 1 \
    --num_workers 1 \
    --sequence_len 12 \
    --limit_samples 10 \
    --mixed_precision
```

### Video Preprocessing
The dataset includes automatic preprocessing:
- File size filtering (skip >100MB videos)
- Frame count validation (minimum 8 frames)
- Automatic downscaling for large videos
- Random cropping for consistent dimensions

## Testing and Evaluation

### Test Script
The optimized test script handles multiple videos efficiently:

```bash
python test_finetuned_model.py
```

Features:
- Automatic video discovery in training directory
- Memory-efficient loading with size limits
- Batch processing with memory cleanup
- Individual output files for each video

### Output Structure
```
test_outputs/
├── output_video1_pred.mp4
├── output_video2_pred.mp4
└── ...
```

## Troubleshooting

### Common Issues

#### 1. DataLoader Worker Killed
**Symptoms**: `RuntimeError: DataLoader worker (pid X) is killed by signal: Killed`

**Solutions**:
- Reduce `num_workers` to 1-2
- Reduce `batch_size` to 1-2
- Enable `mixed_precision`
- Filter out large videos (>100MB)

#### 2. Out of Memory (OOM)
**Symptoms**: CUDA out of memory errors

**Solutions**:
- Reduce batch size
- Reduce sequence length
- Reduce number of trajectories
- Enable gradient checkpointing
- Use smaller crop sizes

#### 3. Video Loading Errors
**Symptoms**: Failed to load video errors

**Solutions**:
- Check video format compatibility
- Ensure videos have enough frames
- Verify file permissions
- Try different video codecs

### Debugging Tips

#### 1. Test Dataset Loading
```python
# Test individual video loading
from custom_dataset import CustomVideoDataset

dataset = CustomVideoDataset(video_dir="./data/train", limit_samples=1)
sample, success = dataset[0]
print(f"Success: {success}")
print(f"Video shape: {sample.video.shape}")
```

#### 2. Monitor Memory Usage
```python
import torch
print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

#### 3. Gradual Scaling
Start with 1-2 videos, then gradually increase:
1. Test with 1 video
2. Add 2-3 more videos
3. Scale to full dataset

## Best Practices

### 1. Data Quality
- Use high-quality, stable videos
- Ensure videos have moving objects
- Avoid videos with rapid camera motion
- Include diverse scenes and objects

### 2. Training Strategy
- Start with fewer videos and shorter sequences
- Gradually increase complexity
- Use mixed precision for efficiency
- Monitor loss curves for convergence

### 3. Memory Management
- Clear cache regularly
- Use appropriate batch sizes
- Limit worker processes
- Monitor GPU memory usage

### 4. Evaluation
- Test on diverse videos
- Compare with baseline model
- Monitor tracking accuracy
- Check for overfitting

### 5. Model Selection
- Online model: Real-time tracking, shorter sequences
- Offline model: Better accuracy, longer sequences
- Choose based on your use case

## Advanced Topics

### Custom Loss Functions
You can modify the loss function in `forward_batch()` to add custom losses:

```python
def custom_loss(pred_tracks, gt_tracks, pred_visibility, gt_visibility):
    # Add your custom loss here
    pass
```

### Data Augmentation
Enhance the dataset with augmentations:
- Random cropping
- Color jittering
- Temporal augmentation
- Spatial transformations

### Multi-GPU Training
For larger datasets, use multiple GPUs:
```bash
python -m torch.distributed.launch --nproc_per_node=2 train_custom_dataset.py
```

## Conclusion

This guide provides a comprehensive approach to fine-tuning CoTracker on your personal dataset. The key is to start small and scale gradually, paying attention to memory management and data quality. With proper optimization, you can successfully train on multiple videos while maintaining good performance.

For additional help, refer to the official CoTracker documentation and GitHub repository. 