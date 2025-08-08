#!/usr/bin/env python3
"""
Test script for fine-tuned CoTracker model.
Optimized for multiple videos and memory efficiency.
"""

import os
import torch
import numpy as np
import json
import argparse
from pathlib import Path
from torchvision.io import read_video
from cotracker.models.build_cotracker import build_cotracker
from cotracker.utils.visualizer import Visualizer
from cotracker.models.core.model_utils import get_points_on_a_grid
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def load_video_efficiently(video_path, max_size_mb=50):
    """Load video with memory constraints"""
    try:
        # Check file size
        file_size = Path(video_path).stat().st_size / (1024 * 1024)  # MB
        if file_size > max_size_mb:
            print(f"Video {video_path} is too large ({file_size:.1f}MB), downscaling...")
            return load_and_downscale_video(video_path)
        
        # Load video normally
        rgbs, _, _ = read_video(str(video_path), output_format="TCHW", pts_unit="sec")
        return rgbs
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None

def load_and_downscale_video(video_path, target_height=480):
    """Load and downscale large videos"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Downscale if needed
            h, w = frame.shape[:2]
            if h > target_height:
                scale = target_height / h
                new_w = int(w * scale)
                frame = cv2.resize(frame, (new_w, target_height))
            
            frames.append(frame)
        
        cap.release()
        
        if frames:
            # Convert to tensor format (T, C, H, W)
            frames = np.stack(frames)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
            return frames
        else:
            return None
            
    except Exception as e:
        print(f"Error downscaling video {video_path}: {e}")
        return None

def create_static_reference_grid(video_shape, grid_size, device, model_interp_shape=None):
    """Create a static reference grid that matches the tracked points"""
    H, W = video_shape[3], video_shape[4]  # For (B, T, C, H, W) format
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
    """Calculate tracking deviation from initial positions"""
    B, T, N, _ = tracked_points.shape
    # static_reference has shape (B, 1, N, 2) - expand to match tracked_points
    static_ref_expanded = static_reference.expand(B, T, N, 2)
    error = tracked_points - static_ref_expanded
    if visibility is not None:
        if visibility.dim() == 3:
            visibility = visibility.unsqueeze(-1)
        error = error * visibility
    return error

def plot_noise_analysis(error_data, frame_numbers, save_path='./noise_analysis.png', max_frames=None):
    """Create noise analysis plots"""
    B, T, N, _ = error_data.shape
    x_errors = error_data[0, :, :, 0].cpu().numpy()
    y_errors = error_data[0, :, :, 1].cpu().numpy()
    
    if max_frames is not None and T > max_frames:
        step = T // max_frames
        plot_indices = np.arange(0, T, step)[:max_frames]
        x_errors_plot = x_errors[plot_indices]
        y_errors_plot = y_errors[plot_indices]
        frame_numbers_plot = [frame_numbers[i] for i in plot_indices]
    else:
        x_errors_plot = x_errors
        y_errors_plot = y_errors
        frame_numbers_plot = frame_numbers
    
    # Calculate statistics
    x_std_per_frame = np.std(x_errors, axis=1)  # Standard deviation per frame
    y_std_per_frame = np.std(y_errors, axis=1)
    x_mean_per_frame = np.mean(x_errors, axis=1)
    y_mean_per_frame = np.mean(y_errors, axis=1)
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # X-axis error over time
    for i in range(min(20, N)):  # Plot first 20 tracks
        ax1.plot(frame_numbers_plot, x_errors_plot[:, i], alpha=0.7, linewidth=0.5)
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('X-axis Deviation (pixels)')
    ax1.set_title('X-axis Deviation from Initial Position Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Y-axis error over time
    for i in range(min(20, N)):
        ax2.plot(frame_numbers_plot, y_errors_plot[:, i], alpha=0.7, linewidth=0.5)
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Y-axis Deviation (pixels)')
    ax2.set_title('Y-axis Deviation from Initial Position Over Time')
    ax2.grid(True, alpha=0.3)
    
    # X-axis statistics
    ax3.plot(frame_numbers_plot, x_mean_per_frame[plot_indices] if max_frames is not None and T > max_frames else x_mean_per_frame, 'b-', label='Mean')
    ax3.fill_between(frame_numbers_plot, 
                     (x_mean_per_frame - x_std_per_frame)[plot_indices] if max_frames is not None and T > max_frames else (x_mean_per_frame - x_std_per_frame),
                     (x_mean_per_frame + x_std_per_frame)[plot_indices] if max_frames is not None and T > max_frames else (x_mean_per_frame + x_std_per_frame),
                     alpha=0.3, color='blue', label='±1 Std Dev')
    ax3.set_xlabel('Frame Number')
    ax3.set_ylabel('X-axis Deviation (pixels)')
    ax3.set_title('X-axis Deviation Statistics')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Y-axis statistics
    ax4.plot(frame_numbers_plot, y_mean_per_frame[plot_indices] if max_frames is not None and T > max_frames else y_mean_per_frame, 'r-', label='Mean')
    ax4.fill_between(frame_numbers_plot,
                     (y_mean_per_frame - y_std_per_frame)[plot_indices] if max_frames is not None and T > max_frames else (y_mean_per_frame - y_std_per_frame),
                     (y_mean_per_frame + y_std_per_frame)[plot_indices] if max_frames is not None and T > max_frames else (y_mean_per_frame + y_std_per_frame),
                     alpha=0.3, color='red', label='±1 Std Dev')
    ax4.set_xlabel('Frame Number')
    ax4.set_ylabel('Y-axis Deviation (pixels)')
    ax4.set_title('Y-axis Deviation Statistics')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate overall statistics (matching online_demo.py format)
    overall_stats = {
        'average_x_noise': float(np.mean(x_std_per_frame)),  # Mean of std per frame
        'average_y_noise': float(np.mean(y_std_per_frame)),  # Mean of std per frame
        'max_x_noise': float(np.max(np.abs(x_errors))),      # Max absolute error
        'max_y_noise': float(np.max(np.abs(y_errors))),      # Max absolute error
        'x_noise_range': [float(np.min(x_errors)), float(np.max(x_errors))],
        'y_noise_range': [float(np.min(y_errors)), float(np.max(y_errors))],
        'total_frames': T,
        'total_points': N
    }
    
    return overall_stats

def create_static_vs_tracked_visualization(video_frames, static_reference, tracked_points, save_path):
    """Create visualization comparing static reference vs tracked points for first frame"""
    first_frame = video_frames[0]
    if first_frame.dtype != np.uint8:
        first_frame = (first_frame * 255).astype(np.uint8)
    
    # Convert to PIL Image
    if first_frame.ndim == 3 and first_frame.shape[0] == 3:  # CHW format
        first_frame = first_frame.transpose(1, 2, 0)  # CHW -> HWC
    img = Image.fromarray(first_frame)
    draw = ImageDraw.Draw(img)
    
    # Draw static reference points (blue X) - these are the initial positions
    static_points = static_reference[0, 0].cpu().numpy()  # Shape: (N, 2)
    for point in static_points:
        x, y = int(point[0]), int(point[1])
        size = 5
        draw.line([(x-size, y-size), (x+size, y+size)], fill=(255, 0, 0), width=2)  # Blue X
        draw.line([(x-size, y+size), (x+size, y-size)], fill=(255, 0, 0), width=2)
    
    # Draw tracked points (red circles) - these should be at the same positions initially
    tracked_points_first = tracked_points[0, 0].cpu().numpy()
    for point in tracked_points_first:
        x, y = int(point[0]), int(point[1])
        size = 3
        draw.ellipse([(x-size, y-size), (x+size, y+size)], outline=(0, 255, 0), width=2)  # Green circle
    
    img.save(save_path)
    print(f"Static vs tracked visualization saved to: {save_path}")

def create_full_video_visualization(video_frames, static_reference, tracked_points, save_path):
    """Create full video visualization showing static reference vs tracked points"""
    if len(video_frames) == 0:
        print("No frames to process")
        return
    
    # Get video dimensions
    H, W = video_frames[0].shape[:2] if video_frames[0].ndim == 3 else video_frames[0].shape[1:]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (W, H))
    
    static_points = static_reference[0, 0].cpu().numpy()  # Shape: (N, 2)
    
    for frame_idx, frame in enumerate(video_frames):
        # Convert frame to uint8 if needed
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        if frame.ndim == 3 and frame.shape[0] == 3:  # CHW format
            frame = frame.transpose(1, 2, 0)  # CHW -> HWC
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hi
        # Draw static reference points (blue X) - initial positions
        for point in static_points:
            x, y = int(point[0]), int(point[1])
            size = 5
            cv2.line(frame, (x-size, y-size), (x+size, y+size), (255, 0, 0), 2)  # Blue X
            cv2.line(frame, (x-size, y+size), (x+size, y-size), (255, 0, 0), 2)
        
        # Draw tracked points (green circles) - current positions
        if frame_idx < tracked_points.shape[1]:
            tracked_points_frame = tracked_points[0, frame_idx].cpu().numpy()
            for point in tracked_points_frame:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), 2)  # Green circle
        
        out.write(frame)
    
    out.release()
    print(f"Full video visualization saved to: {save_path}")

def test_finetuned_model():
    """Test the fine-tuned model on multiple videos"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test fine-tuned CoTracker model with noise analysis')
    parser.add_argument('--model_path', type=str, default="./checkpoints/my_finetuned_model/model_cotracker_three_001170.pth",
                       help='Path to the fine-tuned model checkpoint')
    parser.add_argument('--video_dir', type=str, default="./data/test",
                       help='Directory containing test videos')
    parser.add_argument('--output_dir', type=str, default="./test_outputs",
                       help='Output directory for results')
    parser.add_argument('--grid_size', type=int, default=10,
                       help='Grid size for tracking points')
    parser.add_argument('--noise_analysis', action='store_true',
                       help='Enable noise analysis mode')
    parser.add_argument('--max_frames_for_plot', type=int, default=None,
                       help='Maximum frames to show in noise analysis plots')
    parser.add_argument('--max_frames', type=int, default=30000,
                       help='Maximum frames to process per video')
    parser.add_argument('--online', action='store_true',
                       help='Use online tracking mode instead of offline')
    
    args = parser.parse_args()
    
    # Configuration
    model_path = args.model_path
    video_dir = args.video_dir
    output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run training first or check the model path.")
        return
    
    # Load model
    print("Loading fine-tuned model...")
    try:
        if args.online:
            # Use online tracking mode
            from cotracker.predictor import CoTrackerOnlinePredictor
            model = CoTrackerOnlinePredictor(checkpoint=model_path)
            print("Using ONLINE tracking mode")
        else:
            # Use offline tracking mode
            model = build_cotracker(
                checkpoint=model_path,
                offline=False,
                window_len=16
            )
            print("Using OFFLINE tracking mode")
        
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        print("Model loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f"*{ext}"))
        video_files.extend(Path(video_dir).glob(f"*{ext.upper()}"))
    
    video_files = sorted(list(set(video_files)))
    print(f"Found {len(video_files)} videos to test")
    
    # Test each video
    for i, video_path in enumerate(video_files):
        print(f"\nTesting video {i+1}/{len(video_files)}: {video_path.name}")
        
        #try:
        # Load video efficiently
        video = load_video_efficiently(video_path)
        if video is None:
            print(f"Failed to load {video_path.name}, skipping...")
            continue
        
        # Limit sequence length to save memory
        if len(video) > args.max_frames:
            print(f"Truncating video from {len(video)} to {args.max_frames} frames")
            video = video[:args.max_frames]
        
        # Downscale video if too large
        max_h, max_w = 540, 540
        T, C, H, W = video.shape
        if H > max_h or W > max_w:
            video = F.interpolate(video, size=(max_h, max_w), mode='bilinear', align_corners=False)
            print(f"Downscaled video to: {video.shape}")
        
        if args.online:
            # Online tracking mode - process frames incrementally
            print("Processing video in ONLINE mode...")
            
            # Ensure video is on GPU first, then convert to frames
            if torch.cuda.is_available():
                video = video.cuda()
            
            # Convert video frames to list for online processing
            video_frames = video.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
            video_frames = [frame for frame in video_frames]
            
            # Initialize tracking variables
            window_frames = []
            all_tracks = []
            all_visibility = []
            frame_numbers = []
            
            # Process frames incrementally like in online_demo.py
            is_first_step = True
            frame_count = 0
            
            for i, frame in enumerate(video_frames):
                if i % model.step == 0 and i != 0:
                    # Process current window
                    video_chunk = torch.tensor(
                        np.stack(window_frames[-model.step * 2 :]), 
                        device=next(model.parameters()).device  # Use model's device
                    ).float().permute(0, 3, 1, 2)[None]  # (1, T, 3, H, W)
                    
                    #print(f"Processing window {i//model.step}: video_chunk shape: {video_chunk.shape}, device: {video_chunk.device}")
                    
                    pred_tracks, pred_visibility = model(
                        video_chunk,
                        is_first_step=is_first_step,
                        grid_size=args.grid_size,
                        grid_query_frame=0
                    )
                    
                    # Store tracks and visibility for noise analysis
                    if args.noise_analysis and pred_tracks is not None:
                        all_tracks.append(pred_tracks)
                        all_visibility.append(pred_visibility)
                        frame_numbers.extend(range(frame_count, frame_count + pred_tracks.shape[1]))
                    
                    is_first_step = False
                    frame_count += pred_tracks.shape[1] if pred_tracks is not None else 0
                
                window_frames.append(frame)
            
            # Process final window
            if len(window_frames) >= model.step:
                video_chunk = torch.tensor(
                    np.stack(window_frames[-model.step:]), 
                    device=next(model.parameters()).device  # Use model's device
                ).float().permute(0, 3, 1, 2)[None]
                
                pred_tracks, pred_visibility = model(
                    video_chunk,
                    is_first_step=is_first_step,
                    grid_size=args.grid_size,
                    grid_query_frame=0
                )
                
                if args.noise_analysis and pred_tracks is not None:
                    all_tracks.append(pred_tracks)
                    all_visibility.append(pred_visibility)
                    frame_numbers.extend(range(frame_count, frame_count + pred_tracks.shape[1]))
            
            # Concatenate all tracks for visualization and analysis
            if all_tracks:
                pred_tracks = torch.cat(all_tracks, dim=1)  # (B, T_total, N, 2)
                pred_visibility = torch.cat(all_visibility, dim=1)  # (B, T_total, N, 1)
                print(f"Online tracking completed. Total tracks shape: {pred_tracks.shape}")
            else:
                print("No tracks generated in online mode")
                continue
                
        else:
            # Offline tracking mode - process entire video at once
            print("Processing video in OFFLINE mode...")
            
            # Ensure video is on GPU
            if torch.cuda.is_available():
                video = video.cuda()
                video = video.unsqueeze(0)
            
            # Create queries (grid that fits the video size)
            B, T, C, H, W = video.shape
            grid_size = args.grid_size
            ys = torch.linspace(0, H - 1, grid_size)
            xs = torch.linspace(0, W - 1, grid_size)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            yy = yy.flatten()
            xx = xx.flatten()
            t = torch.zeros_like(yy)  # all queries on frame 0
            queries = torch.stack([t, yy, xx], dim=1).unsqueeze(0).to(video.device)  # [1, grid_size*grid_size, 3]
            print(f"Generated grid queries to fit video: {queries.shape}")
            
            if torch.cuda.is_available():
                queries = queries.cuda()
            
            print(f"Video shape: {video.shape}")
            print(f"Queries shape: {queries.shape}")
            
            # Run model with custom queries
            with torch.no_grad():
                pred_tracks, pred_visibility, conf_predicted, train_data = model(video, queries=queries, iters=10)
        
        # Threshold visibility to get boolean mask
        visibility_thresh = 0.1
        pred_visibility_bool = (pred_visibility > visibility_thresh)
        
        # Create output filename
        output_filename = f"output_{video_path.stem}_pred"
        output_video = os.path.join(output_dir, f"{output_filename}.mp4")
        
        # Prepare video for visualization
        if args.online:
            # For online mode, create video tensor from frames
            video_for_viz = torch.tensor(np.stack(video_frames), device=pred_tracks.device).permute(0, 3, 1, 2)[None]
        else:
            # For offline mode, use the existing video tensor
            video_for_viz = video
        
        # Visualize
        vis = Visualizer(save_dir=output_dir, pad_value=120, linewidth=3)
        vis.visualize(video_for_viz, pred_tracks, pred_visibility_bool, filename=output_filename)
        
        print(f"Visualization saved to: {output_video}")
        
        # Noise analysis if enabled
        if args.noise_analysis:
            print(f"Performing noise analysis for {video_path.name}...")
            
            # Create noise analysis output folder
            video_name = video_path.stem
            noise_output_folder = os.path.join(output_dir, f"noise_analysis_{video_name}")
            os.makedirs(noise_output_folder, exist_ok=True)
            
            # Create static reference based on initial tracked points from first frame
            device = video.device
            # Use the tracked points from the first frame as the static reference
            static_reference = pred_tracks[:, 0:1, :, :].clone()  # Shape: (B, 1, N, 2)
            print(f"Static reference created from initial tracked points: {static_reference.shape}")
            print(f"Initial tracked points X range: [{static_reference[0, 0, :, 0].min():.2f}, {static_reference[0, 0, :, 0].max():.2f}]")
            print(f"Initial tracked points Y range: [{static_reference[0, 0, :, 1].min():.2f}, {static_reference[0, 0, :, 1].max():.2f}]")
            
            # Convert video frames for visualization
            if args.online:
                # For online mode, video_frames is already a list of numpy arrays
                video_frames_np = np.stack(video_frames)
            else:
                # For offline mode, convert video tensor to numpy
                video_frames_np = video[0].permute(0, 2, 3, 1).cpu().numpy()
            
            # Create visualizations
            create_static_vs_tracked_visualization(
                video_frames_np, static_reference, pred_tracks, 
                f"{noise_output_folder}/static_vs_tracked_comparison.png"
            )
            create_full_video_visualization(
                video_frames_np, static_reference, pred_tracks, 
                f"{noise_output_folder}/video_with_static_reference.mp4"
            )
            
            # Calculate tracking noise
            error_data = calculate_tracking_noise(pred_tracks, static_reference, pred_visibility)
            if args.online:
                # For online mode, use the collected frame numbers
                frame_numbers = frame_numbers
            else:
                # For offline mode, create sequential frame numbers
                frame_numbers = list(range(pred_tracks.shape[1]))
            
            # Create noise analysis plots
            noise_stats = plot_noise_analysis(
                error_data, frame_numbers, 
                f"{noise_output_folder}/noise_analysis.png", 
                args.max_frames_for_plot
            )
            
            # Save error data and statistics
            np.save(f"{noise_output_folder}/error_data.npy", error_data.cpu().numpy())
            with open(f"{noise_output_folder}/noise_statistics.json", "w") as f:
                json.dump(noise_stats, f, indent=2)
            
            print(f"Noise analysis completed for {video_path.name}")
            print(f"Results saved to: {noise_output_folder}")
            
            # Copy the main output video to noise analysis folder
            import shutil
            main_output_video = os.path.join(output_dir, f"{output_filename}.mp4")
            if os.path.exists(main_output_video):
                shutil.copy2(main_output_video, f"{noise_output_folder}/tracking_output.mp4")
        
        # Clear memory
        del video, pred_tracks, pred_visibility, pred_visibility_bool
        if not args.online:
            del queries  # queries only exists in offline mode
        if args.noise_analysis:
            del error_data, static_reference
        torch.cuda.empty_cache()
                
        # except Exception as e:
        #     print(f"Error processing {video_path.name}: {e}")
        #     torch.cuda.empty_cache()
        #     continue
    
    print("\nTesting completed!")

if __name__ == "__main__":
    test_finetuned_model() 