# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from cotracker.models.core.model_utils import get_points_on_a_grid

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor


DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

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

def create_static_vs_tracked_visualization(video_frames, static_reference, tracked_points, save_path="./static_vs_tracked_comparison.png"):
    """Create a visualization showing static reference points vs tracked points from first frame"""
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    
    # Get the first frame
    first_frame = video_frames[0]  # Shape: (H, W, 3)
    
    # Convert to PIL Image for drawing
    pil_image = Image.fromarray(first_frame)
    draw = ImageDraw.Draw(pil_image)
    
    # Draw tracked points from first frame (larger, red)
    tracked_first_frame = tracked_points[0, 0]  # Shape: (N, 2)
    for i in range(tracked_first_frame.shape[0]):
        x, y = tracked_first_frame[i, 0].item(), tracked_first_frame[i, 1].item()
        # Draw larger red circles for tracked points
        draw.ellipse([x-5, y-5, x+5, y+5], fill=(255, 0, 0), outline=(255, 0, 0))
    
    # Draw static reference points (smaller, blue)
    static_points = static_reference[0]  # Shape: (N, 2)
    for i in range(static_points.shape[0]):
        x, y = static_points[i, 0].item(), static_points[i, 1].item()
        # Draw smaller blue circles for static points
        draw.ellipse([x-3, y-3, x+3, y+3], fill=(0, 0, 255), outline=(0, 0, 255))
    
    # Save the image
    pil_image.save(save_path)
    print(f"Static vs tracked comparison saved to {save_path}")
    
    # Also create a matplotlib version with legend
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(first_frame)
    
    # Plot tracked points
    tracked_x = tracked_first_frame[:, 0].cpu().numpy()
    tracked_y = tracked_first_frame[:, 1].cpu().numpy()
    ax.scatter(tracked_x, tracked_y, c='red', s=100, alpha=0.8, label='Tracked Points (Frame 0)', marker='o')
    
    # Plot static points
    static_x = static_points[:, 0].cpu().numpy()
    static_y = static_points[:, 1].cpu().numpy()
    ax.scatter(static_x, static_y, c='blue', s=50, alpha=0.8, label='Static Reference Points', marker='x')
    
    ax.set_title('Static Reference Points vs Tracked Points (Frame 0)')
    ax.legend()
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_matplotlib.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matplotlib comparison saved to {save_path.replace('.png', '_matplotlib.png')}")

def create_static_reference_grid(video_shape, grid_size, device, model_interp_shape=None):
    """Create a static reference grid that doesn't move"""
    H, W = video_shape[2], video_shape[3]  # Height, Width
    
    # Use model's interpolation shape if provided, otherwise use video shape
    if model_interp_shape is not None:
        grid_shape = model_interp_shape
    else:
        grid_shape = [H, W]
    
    static_grid = get_points_on_a_grid(grid_size, grid_shape, device=device)
    
    # If we used model's interpolation shape, scale back to original video dimensions
    if model_interp_shape is not None:
        static_grid *= static_grid.new_tensor(
            [(W - 1) / (model_interp_shape[1] - 1), (H - 1) / (model_interp_shape[0] - 1)]
        )
    # Note: get_points_on_a_grid already returns coordinates in the correct range (0 to W-1, 0 to H-1)
    # so no additional scaling is needed when using video shape directly
    
    return static_grid  # Shape: (1, grid_size*grid_size, 2)

def calculate_tracking_noise(tracked_points, static_reference, visibility):
    """Calculate the difference between tracked points and static reference"""
    # tracked_points: (B, T, N, 2) - tracked positions over time
    # static_reference: (1, N, 2) - static reference positions
    # visibility: (B, T, N) or (B, T, N, 1) - visibility mask
    
    B, T, N, _ = tracked_points.shape
    
    # Expand static reference to match tracked points shape
    static_ref_expanded = static_reference.expand(B, T, N, 2)
    
    # Calculate difference (error)
    error = tracked_points - static_ref_expanded  # (B, T, N, 2)
    
    # Apply visibility mask
    if visibility is not None:
        # Ensure visibility has the right shape for broadcasting
        if visibility.dim() == 3:  # (B, T, N)
            visibility = visibility.unsqueeze(-1)  # (B, T, N, 1)
        error = error * visibility  # Zero out errors for invisible points
    
    return error  # Shape: (B, T, N, 2)

def plot_noise_analysis(error_data, frame_numbers, save_path="./noise_analysis.png", max_frames=None):
    """Plot X and Y noise over time"""
    B, T, N, _ = error_data.shape
    
    # Extract X and Y errors
    x_errors = error_data[0, :, :, 0].cpu().numpy()  # (T, N)
    y_errors = error_data[0, :, :, 1].cpu().numpy()  # (T, N)
    
    # Limit frames for plotting if specified
    if max_frames is not None and T > max_frames:
        # Sample frames evenly across the timeline
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
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot X errors
    for i in range(N):
        ax1.plot(frame_numbers_plot, x_errors_plot[:, i], alpha=0.7, linewidth=1)
    ax1.set_title("X-axis Tracking Noise")
    ax1.set_xlabel("Frame Number")
    ax1.set_ylabel("X Error (pixels)")
    ax1.grid(True, alpha=0.3)
    
    # Plot Y errors
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
    
    # Calculate and print statistics (using all data, not just plotted data)
    x_std = np.std(x_errors, axis=1)  # Standard deviation per frame
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./michael_vids/tissue_static_4.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
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
        "--noise_analysis",
        action="store_true",
        help="Perform noise analysis with static reference grid",
    )
    parser.add_argument(
        "--max_frames_for_plot",
        type=int,
        default=500,
        help="Maximum number of frames to show in noise plots for better visualization",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(DEFAULT_DEVICE)

    window_frames = []
    all_tracks = []
    all_visibility = []
    frame_numbers = []

    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    frame_count = 0
    
    for i, frame in enumerate(
        iio.imiter(
            args.video_path,
            plugin="FFMPEG",
        )
    ):
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
            )
            
            # Debug output
            print(f"Frame {i}: pred_tracks shape: {pred_tracks.shape if pred_tracks is not None else 'None'}")
            print(f"Frame {i}: pred_visibility shape: {pred_visibility.shape if pred_visibility is not None else 'None'}")
            
            # Store tracks and visibility for noise analysis
            if args.noise_analysis and pred_tracks is not None:
                all_tracks.append(pred_tracks)
                all_visibility.append(pred_visibility)
                frame_numbers.extend(range(frame_count, frame_count + pred_tracks.shape[1]))
            
            is_first_step = False
            frame_count += pred_tracks.shape[1] if pred_tracks is not None else 0
            
        window_frames.append(frame)
    
    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
    )
    
    # Debug output for final step
    print(f"Final step: pred_tracks shape: {pred_tracks.shape if pred_tracks is not None else 'None'}")
    print(f"Final step: pred_visibility shape: {pred_visibility.shape if pred_visibility is not None else 'None'}")
    
    # Store final tracks
    if args.noise_analysis and pred_tracks is not None:
        all_tracks.append(pred_tracks)
        all_visibility.append(pred_visibility)
        frame_numbers.extend(range(frame_count, frame_count + pred_tracks.shape[1]))

    print("Tracks are computed")
    print(f"Total tracks collected: {len(all_tracks)}")
    print(f"Total frame numbers: {len(frame_numbers)}")

    # Perform noise analysis if requested
    if args.noise_analysis and all_tracks:
        print("Performing noise analysis...")
        
        # Create output folder based on video name
        video_name = os.path.basename(args.video_path).split('.')[0]
        output_folder = f"./noise_analysis_{video_name}"
        os.makedirs(output_folder, exist_ok=True)
        print(f"Creating output folder: {output_folder}")
        
        # Concatenate all tracks
        all_tracks_tensor = torch.cat(all_tracks, dim=1)  # (B, T_total, N, 2)
        all_visibility_tensor = torch.cat(all_visibility, dim=1)  # (B, T_total, N, 1)
        
        print(f"Concatenated tracks shape: {all_tracks_tensor.shape}")
        print(f"Concatenated visibility shape: {all_visibility_tensor.shape}")
        
        # Create static reference grid using the actual video dimensions
        H, W = window_frames[0].shape[0], window_frames[0].shape[1]  # Height, Width from first frame
        static_reference = create_static_reference_grid([1, 1, H, W], args.grid_size, DEFAULT_DEVICE, model.interp_shape)
        
        # Debug: Check coordinate ranges
        print(f"Model interp_shape: {model.interp_shape}")
        print(f"Video dimensions: {H} x {W}")
        print(f"Static reference X range: [{static_reference[0, :, 0].min():.2f}, {static_reference[0, :, 0].max():.2f}]")
        print(f"Static reference Y range: [{static_reference[0, :, 1].min():.2f}, {static_reference[0, :, 1].max():.2f}]")
        print(f"Initial tracks X range: [{all_tracks_tensor[0, 0, :, 0].min():.2f}, {all_tracks_tensor[0, 0, :, 0].max():.2f}]")
        print(f"Initial tracks Y range: [{all_tracks_tensor[0, 0, :, 1].min():.2f}, {all_tracks_tensor[0, 0, :, 1].max():.2f}]")
        
        # Create visualization comparing static reference vs tracked points
        create_static_vs_tracked_visualization(
            window_frames, 
            static_reference, 
            all_tracks_tensor,
            f"{output_folder}/static_vs_tracked_comparison.png"
        )
        
        # Create a video visualization showing static reference points vs tracked points throughout the video
        create_full_video_visualization(
            window_frames,
            static_reference,
            all_tracks_tensor,
            f"{output_folder}/video_with_static_reference.mp4"
        )
        
        # Calculate tracking noise
        error_data = calculate_tracking_noise(all_tracks_tensor, static_reference, all_visibility_tensor)
        
        # Plot noise analysis
        noise_stats = plot_noise_analysis(error_data, frame_numbers, f"{output_folder}/noise_analysis.png", args.max_frames_for_plot)
        
        # Save error data for further analysis
        np.save(f"{output_folder}/error_data.npy", error_data.cpu().numpy())
        print(f"Error data saved to {output_folder}/error_data.npy")
        
        # Save noise statistics as JSON
        import json
        with open(f"{output_folder}/noise_statistics.json", 'w') as f:
            json.dump(noise_stats, f, indent=2)
        print(f"Noise statistics saved to {output_folder}/noise_statistics.json")
        
        print(f"All analysis results saved to folder: {output_folder}")
    elif args.noise_analysis:
        print("Warning: No tracks collected for noise analysis")

    # save a video with predicted tracks
    if pred_tracks is not None:
        seq_name = args.video_path.split("/")[-1]
        video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
            0, 3, 1, 2
        )[None]
        
        # Save video in output folder if noise analysis was performed
        if args.noise_analysis and all_tracks:
            video_name = os.path.basename(args.video_path).split('.')[0]
            output_folder = f"./noise_analysis_{video_name}"
            vis = Visualizer(save_dir=output_folder, pad_value=120, linewidth=3)
            vis.visualize(
                video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame
            )
            print(f"Video with tracks saved to {output_folder}/video.mp4")
        else:
            vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
            vis.visualize(
                video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame
            )
    else:
        print("Warning: No tracks available for video visualization")
