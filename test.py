import torch
import torch.nn.functional as F
import numpy as np
import cv2
# Download the video
url = './data/test/97_109_top_IFBS_ENDOSCOPE-part0005.mp4'

import imageio.v3 as iio
frames = iio.imread(url, plugin="FFMPEG")  # plugin="pyav"

# Downsample frames if too large
max_h, max_w = 540, 540
h, w = frames.shape[1:3]
if h > max_h or w > max_w:
    # Compute new size while preserving aspect ratio
    scale = min(max_h / h, max_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    # Ensure frames are uint8 for cv2
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    frames_resized = []
    for f in frames:
        frames_resized.append(cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR))
    frames = np.stack(frames_resized)
    print(f"Downsampled video to: {frames.shape}")

# Ensure frames is a numpy array before converting to torch tensor
if isinstance(frames, np.ndarray):
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float()
    device = 'cuda'
    video = video.to(device)  # B T C H W
else:
    raise ValueError("Frames is not a numpy array after processing.")

grid_size = 10
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)

# Run Online CoTracker, the same model with a different API:
# Initialize online processing
cotracker(video_chunk=video, is_first_step=True, grid_size=grid_size)  

# Process the video
for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
    pred_tracks, pred_visibility = cotracker(
        video_chunk=video[:, ind : ind + cotracker.step * 2]
    )  # B T N 2,  B T N 1


from cotracker.utils.visualizer import Visualizer

vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
vis.visualize(video, pred_tracks, pred_visibility)