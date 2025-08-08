# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor
from cotracker.models.core.model_utils import get_points_on_a_grid

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# if DEFAULT_DEVICE == "mps":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def create_static_reference_grid(video_shape, grid_size, device, model_interp_shape=None):
    """Create a static reference grid that doesn't move"""
    # video_shape is (B, T, C, H, W)
    H, W = video_shape[3], video_shape[4]  # Height, Width are in the last two dimensions
    if model_interp_shape is not None:
        grid_shape = model_interp_shape
    else:
        grid_shape = [H, W]
    static_grid = get_points_on_a_grid(grid_size, grid_shape, device=device)
    if model_interp_shape is not None:
        static_grid *= static_grid.new_tensor(
            [(W - 1) / (model_interp_shape[1] - 1), (H - 1) / (model_interp_shape[0] - 1)]
        )
    return static_grid

def calculate_tracking_noise(tracked_points, static_reference, visibility):
    """Calculate the difference between tracked points and static reference"""
    B, T, N, _ = tracked_points.shape
    static_ref_expanded = static_reference.expand(B, T, N, 2)
    error = tracked_points - static_ref_expanded
    if visibility is not None:
        if visibility.dim() == 3:
            visibility = visibility.unsqueeze(-1)
        error = error * visibility
    return error

def plot_noise_analysis(error_data, frame_numbers, save_path="./noise_analysis.png", max_frames=None):
    """Plot X and Y noise over time"""
    B, T, N, _ = error_data.shape
    x_errors = error_data[0, :, :, 0].cpu().numpy()
    y_errors = error_data[0, :, :, 1].cpu().numpy()
    if max_frames is not None and T > max_frames:
        step = T // max_frames
        plot_indices = np.arange(0, T, step)[:max_frames]
        x_errors_plot = x_errors[plot_indices]
        y_errors_plot = y_errors[plot_indices]
        frame_numbers_plot = [frame_numbers[i] for i in plot_indices]
        print(f"Plotting {len(plot_indices)} frames out of {T} total frames for better visualization")
        print(f"Frame range: {frame_numbers_plot[0]} to {frame_numbers_plot[-1]}")
    else:
        x_errors_plot = x_errors
        y_errors_plot = y_errors
        frame_numbers_plot = frame_numbers
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    for i in range(N):
        ax1.plot(frame_numbers_plot, x_errors_plot[:, i], alpha=0.7, linewidth=1)
    ax1.set_title("X-axis Tracking Noise")
    ax1.set_xlabel("Frame Number")
    ax1.set_ylabel("X Error (pixels)")
    ax1.grid(True, alpha=0.3)
    for i in range(N):
        ax2.plot(frame_numbers_plot, y_errors_plot[:, i], alpha=0.7, linewidth=1)
    ax2.set_title("Y-axis Tracking Noise")
    ax2.set_xlabel("Frame Number")
    ax2.set_ylabel("Y Error (pixels)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Noise analysis plot saved to {save_path}")
    x_std = np.std(x_errors, axis=1)
    y_std = np.std(y_errors, axis=1)
    print(f"Average X noise (std): {np.mean(x_std):.4f} pixels")
    print(f"Average Y noise (std): {np.mean(y_std):.4f} pixels")
    print(f"Max X noise: {np.max(np.abs(x_errors)):.4f} pixels")
    print(f"Max Y noise: {np.max(np.abs(y_errors)):.4f} pixels")
    return {
        'average_x_noise': float(np.mean(x_std)),
        'average_y_noise': float(np.mean(y_std)),
        'max_x_noise': float(np.max(np.abs(x_errors))),
        'max_y_noise': float(np.max(np.abs(y_errors))),
        'x_noise_range': [float(np.min(x_errors)), float(np.max(x_errors))],
        'y_noise_range': [float(np.min(y_errors)), float(np.max(y_errors))],
        'total_frames': T,
        'total_points': N
    }

def create_static_vs_tracked_visualization(video_frames, static_reference, tracked_points, save_path="./static_vs_tracked_comparison.png"):
    """Create a visualization showing static reference points vs tracked points from first frame"""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # Get first frame
    first_frame = video_frames[0]
    print(f"First frame shape: {first_frame.shape}")
    if isinstance(first_frame, torch.Tensor):
        first_frame = first_frame.cpu().numpy()
        print(f"First frame numpy shape: {first_frame.shape}")
    
    # Convert to PIL Image - video_frames should be (T, H, W, C) format
    if len(first_frame.shape) == 3 and first_frame.shape[2] == 3:  # HWC format
        # Already in correct format
        print(f"Frame in HWC format: {first_frame.shape}")
    elif first_frame.shape[0] == 3:  # CHW format
        first_frame = np.transpose(first_frame, (1, 2, 0))
        print(f"After transpose (CHW->HWC): {first_frame.shape}")
    elif len(first_frame.shape) == 4:  # BCHW format
        first_frame = first_frame[0]  # Take first batch
        first_frame = np.transpose(first_frame, (1, 2, 0))
        print(f"After batch and transpose: {first_frame.shape}")
    first_frame = (first_frame * 255).astype(np.uint8)
    print(f"After scaling to uint8: {first_frame.shape}")
    img = Image.fromarray(first_frame)
    draw = ImageDraw.Draw(img)
    
    # Get static reference points
    static_points = static_reference[0].cpu().numpy()
    tracked_first_points = tracked_points[0, 0].cpu().numpy()
    
    # Draw static reference points as blue "X" marks
    for i in range(static_points.shape[0]):
        x, y = int(static_points[i, 0]), int(static_points[i, 1])
        if 0 <= x < img.width and 0 <= y < img.height:
            # Draw blue "X"
            draw.line([(x-5, y-5), (x+5, y+5)], fill=(255, 0, 0), width=2)  # Blue
            draw.line([(x-5, y+5), (x+5, y-5)], fill=(255, 0, 0), width=2)  # Blue
    
    # Draw tracked points as red circles
    for i in range(tracked_first_points.shape[0]):
        x, y = int(tracked_first_points[i, 0]), int(tracked_first_points[i, 1])
        if 0 <= x < img.width and 0 <= y < img.height:
            # Draw red circle
            draw.ellipse([x-5, y-5, x+5, y+5], fill=(0, 0, 255), outline=(0, 0, 255))  # Red
    
    # Add title and legend
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "Static Reference Points vs Tracked Points - Frame 0", fill=(255, 255, 255), font=font)
    draw.text((10, img.height - 60), "Blue X: Static Reference", fill=(255, 0, 0), font=font)
    draw.text((10, img.height - 40), "Red Circle: Tracked Points", fill=(0, 0, 255), font=font)
    
    img.save(save_path)
    print(f"Static vs tracked comparison saved to {save_path}")

def create_full_video_visualization(video_frames, static_reference, tracked_points, save_path="./video_with_static_reference.mp4"):
    """Create a video visualization showing static reference points vs tracked points throughout the video"""
    import cv2
    import numpy as np
    
    # Get video dimensions
    H, W = video_frames[0].shape[0], video_frames[0].shape[1]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 10.0, (W, H))
    
    # Get static reference points
    static_points = static_reference[0].cpu().numpy()  # Shape: (N, 2)
    
    print(f"Creating video visualization with {len(video_frames)} frames...")
    
    for frame_idx, frame in enumerate(video_frames):
        # Convert frame to BGR for OpenCV
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if frame.shape[0] == 3:  # CHW format
            frame = np.transpose(frame, (1, 2, 0))
        frame = (frame * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Get tracked points for this frame (if available)
        if frame_idx < tracked_points.shape[1]:
            tracked_frame_points = tracked_points[0, frame_idx].cpu().numpy()  # Shape: (N, 2)
        else:
            # If we don't have tracked points for this frame, use static points
            tracked_frame_points = static_points
        
        # Draw static reference points as blue "X" marks
        for i in range(static_points.shape[0]):
            x, y = int(static_points[i, 0]), int(static_points[i, 1])
            if 0 <= x < W and 0 <= y < H:
                # Draw blue "X" mark
                cv2.line(frame_bgr, (x-5, y-5), (x+5, y+5), (255, 0, 0), 2)  # Blue diagonal
                cv2.line(frame_bgr, (x-5, y+5), (x+5, y-5), (255, 0, 0), 2)  # Blue diagonal
        
        # Draw tracked points as red circles
        for i in range(tracked_frame_points.shape[0]):
            x, y = int(tracked_frame_points[i, 0]), int(tracked_frame_points[i, 1])
            if 0 <= x < W and 0 <= y < H:
                # Draw red circle
                cv2.circle(frame_bgr, (x, y), 5, (0, 0, 255), -1)  # Red filled circle
        
        # Add frame number and title
        cv2.putText(frame_bgr, f"Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_bgr, "Blue X: Static Reference, Red Circle: Tracked", (10, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to video
        out.write(frame_bgr)
        
        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}/{len(video_frames)}")
    
    # Release video writer
    out.release()
    print(f"Video visualization saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--mask_path",
        default="./assets/apple_mask.png",
        help="path to a segmentation mask",
    )
    parser.add_argument(
        "--checkpoint",
        # default="./checkpoints/cotracker.pth",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    parser.add_argument(
        "--use_v2_model",
        action="store_true",
        help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Pass it if you would like to use the offline model, in case of online don't pass it",
    )
    parser.add_argument(
        "--noise_analysis",
        action="store_true",
        help="Perform noise analysis on static video",
    )
    parser.add_argument(
        "--max_frames_for_plot",
        type=int,
        default=None,
        help="Maximum frames to show in noise analysis plots (for better visualization)",
    )

    args = parser.parse_args()

    # load the input video frame by frame
    video = read_video_from_path(args.video_path)
    print(f"Original video shape: {video.shape}")
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    print(f"Processed video shape: {video.shape}")
    segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
    segm_mask = torch.from_numpy(segm_mask)[None, None]

    if args.checkpoint is not None:
        if args.use_v2_model:
            model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=args.use_v2_model)
        else:
            if args.offline:
                window_len = 60
            else:
                window_len = 16
            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=args.use_v2_model,
                offline=args.offline,
                window_len=window_len,
            )
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)

    # Check if we're using an online model
    if 'Online' in str(type(model)):  # Online model
        pred_tracks, pred_visibility = model(
            video,
            grid_size=args.grid_size,
            grid_query_frame=args.grid_query_frame,
            # segm_mask=segm_mask
        )
    else:  # Offline model
        pred_tracks, pred_visibility = model(
            video,
            grid_size=args.grid_size,
            grid_query_frame=args.grid_query_frame,
            backward_tracking=args.backward_tracking,
            # segm_mask=segm_mask
        )
    print("computed")

    # Perform noise analysis if requested
    if args.noise_analysis:
        print("Performing noise analysis...")
        
        # Create output folder
        video_name = args.video_path.split("/")[-1].split(".")[0]
        output_folder = f"./noise_analysis_{video_name}"
        os.makedirs(output_folder, exist_ok=True)
        print(f"Creating output folder: {output_folder}")
        
        # Get video dimensions
        H, W = video.shape[3], video.shape[4]  # Height and Width are in the last two dimensions
        print(f"Video dimensions: {H} x {W}")
        
        # Create static reference grid
        static_reference = create_static_reference_grid(
            video.shape, 
            args.grid_size, 
            DEFAULT_DEVICE, 
            model.interp_shape
        )
        
        # Print coordinate ranges for debugging
        static_points = static_reference[0].cpu().numpy()
        initial_tracks = pred_tracks[0, 0].cpu().numpy()
        print(f"Static reference X range: [{static_points[:, 0].min():.2f}, {static_points[:, 0].max():.2f}]")
        print(f"Static reference Y range: [{static_points[:, 1].min():.2f}, {static_points[:, 1].max():.2f}]")
        print(f"Initial tracks X range: [{initial_tracks[:, 0].min():.2f}, {initial_tracks[:, 0].max():.2f}]")
        print(f"Initial tracks Y range: [{initial_tracks[:, 1].min():.2f}, {initial_tracks[:, 1].max():.2f}]")
        
        # Create static vs tracked visualization
        # Convert video to proper format: (B, T, C, H, W) -> (T, H, W, C)
        print(f"Video tensor shape: {video.shape}")
        print(f"Video[0] tensor shape: {video[0].shape}")
        video_frames = video[0].permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
        print(f"Video frames shape: {video_frames.shape}")
        create_static_vs_tracked_visualization(
            video_frames,
            static_reference,
            pred_tracks,
            f"{output_folder}/static_vs_tracked_comparison.png"
        )
        
        # Create a video visualization showing static reference points vs tracked points throughout the video
        create_full_video_visualization(
            video_frames,
            static_reference,
            pred_tracks,
            f"{output_folder}/video_with_static_reference.mp4"
        )
        
        # Calculate tracking noise
        error_data = calculate_tracking_noise(pred_tracks, static_reference, pred_visibility)
        
        # Create frame numbers for plotting
        frame_numbers = list(range(pred_tracks.shape[1]))
        
        # Plot noise analysis
        noise_stats = plot_noise_analysis(
            error_data, 
            frame_numbers, 
            f"{output_folder}/noise_analysis.png",
            args.max_frames_for_plot
        )
        
        # Save error data and statistics
        np.save(f"{output_folder}/error_data.npy", error_data.cpu().numpy())
        with open(f"{output_folder}/noise_statistics.json", "w") as f:
            json.dump(noise_stats, f, indent=2)
        
        print(f"Error data saved to {output_folder}/error_data.npy")
        print(f"Noise statistics saved to {output_folder}/noise_statistics.json")
        print(f"All analysis results saved to folder: {output_folder}")

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video,
        pred_tracks,
        pred_visibility,
        query_frame=0 if args.backward_tracking else args.grid_query_frame,
    )
    
    # If noise analysis was performed, also save the video to the analysis folder
    if args.noise_analysis:
        import shutil
        video_name = args.video_path.split("/")[-1].split(".")[0]
        output_folder = f"./noise_analysis_{video_name}"
        if os.path.exists("./saved_videos/video.mp4"):
            shutil.copy("./saved_videos/video.mp4", f"{output_folder}/video.mp4")
            print(f"Video with tracks saved to {output_folder}/video.mp4")
