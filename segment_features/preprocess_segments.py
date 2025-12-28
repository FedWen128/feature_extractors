import argparse
import os
import sys
import json

# Add the parent directory to the Python path so we can import omnivore_transforms
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import av
from pytorchvideo.data.encoded_video import EncodedVideo
import concurrent.futures
import logging
from tqdm import tqdm


# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocessing script to decode and cache video segments.")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of videos to process")
    parser.add_argument("--num_threads", type=int, default=10, help="Number of threads for parallel video decoding")
    parser.add_argument("--cache_dir", type=str, default="../data/cache/segments", help="Directory to save cached segments")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for processing (cuda or cpu)")
    return parser.parse_args()


# Video Preprocessing
class SegmentPreprocessor:
    def __init__(self, cache_dir, num_threads=10, device="cpu"):
        self.cache_dir = cache_dir
        self.num_threads = num_threads
        self.fps = 30
        self.num_frames_per_feature = 30
        self.device = torch.device(device if device != "cpu" else "cpu")
        
        # Log device information
        if self.device.type == "cuda":
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        else:
            logger.info("Using CPU for processing")
        
    def preprocess_video(self, video_name, video_directory_path):
        segment_size = self.fps / self.num_frames_per_feature
        video_path = os.path.join(video_directory_path, f"{video_name}.mp4" if "mp4" not in video_name else video_name)

        # Create video-specific cache directory
        video_cache_dir = os.path.join(self.cache_dir, video_name)
        metadata_file = os.path.join(video_cache_dir, "metadata.json")
        
        # Check if already cached
        if os.path.exists(metadata_file):
            logger.info(f"Skipping video (already cached): {video_name}")
            return
        
        os.makedirs(video_cache_dir, exist_ok=True)

        try:
            container = av.open(video_path)
            video_stream = next(s for s in container.streams if s.type == 'video')
            video_duration = float(container.duration / av.time_base)
            
            # Enable GPU hardware decoding if available
            if self.device.type == "cuda":
                video_stream.codec_context.options = {"hwaccel": "cuda", "hwaccel_device": "0"}
        except Exception as e:
            logger.error(f"Failed to load video {video_name}: {e}")
            return

        logger.info(f"video: {video_name} video_duration: {video_duration} s (Device: {self.device})")
        segment_end = max(video_duration - segment_size + 1, 1)
        
        segments = []
        segment_index = 0
        frame_buffer = []
        target_frames = int(self.num_frames_per_feature)
        current_segment_start = 0.0
        
        try:
            # Seek to start of video
            container.seek(0)
            
            for frame in container.decode(video_stream):
                frame_time = float(frame.pts * video_stream.time_base)
                
                # Convert frame to tensor on GPU if available
                frame_array = frame.to_ndarray(format='rgb24')
                frame_tensor = torch.from_numpy(frame_array).to(self.device)
                frame_buffer.append(frame_tensor)
                
                # Check if we have enough frames for a segment
                if len(frame_buffer) >= target_frames:
                    if frame_time >= current_segment_start + segment_size:
                        # Stack frames and move back to CPU for saving
                        segment_video = torch.stack(frame_buffer[:target_frames]).cpu().numpy()
                        
                        # Save segment
                        segment_path = os.path.join(video_cache_dir, f"segment_{segment_index:06d}.npy")
                        np.save(segment_path, segment_video)
                        
                        segments.append({
                            "index": segment_index,
                            "start_time": float(current_segment_start),
                            "end_time": float(current_segment_start + segment_size),
                            "path": segment_path
                        })
                        
                        segment_index += 1
                        current_segment_start += segment_size
                        frame_buffer = []
                        
                        # Stop if we've reached the end
                        if current_segment_start >= segment_end:
                            break
                
                if frame_time >= segment_end:
                    break
        except Exception as e:
            logger.error(f"Failed to process video {video_name}: {e}")
            container.close()
            return
        
        container.close()

        # Save metadata
        metadata = {
            "video_name": video_name,
            "video_duration": video_duration,
            "segment_size": segment_size,
            "total_segments": len(segments),
            "segments": segments,
            "device": str(self.device)
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Finished preprocessing video: {video_name} with {len(segments)} segments")


def main():
    video_files_path = "../data/video/"
    
    preprocessor = SegmentPreprocessor(cache_dir=cache_dir, num_threads=num_threads, device=device)

    mp4_files = [file for file in os.listdir(video_files_path) if file.endswith(".mp4")]
    
    # Limit number of videos if specified
    if max_videos is not None:
        mp4_files = mp4_files[:max_videos]
    
    logger.info(f"Preprocessing {len(mp4_files)} videos")

    num_threads_to_use = min(num_threads, 4)  # Limit to 4 threads for heavy I/O
    with concurrent.futures.ThreadPoolExecutor(num_threads_to_use) as executor:
        list(
            tqdm(
                executor.map(
                    lambda file: preprocessor.preprocess_video(file, video_files_path), mp4_files
                ), total=len(mp4_files)
            )
        )
    
    logger.info(f"Preprocessing complete! Cache stored in {cache_dir}")


if __name__ == "__main__":
    args = parse_arguments()
    max_videos = args.max_videos
    num_threads = args.num_threads
    cache_dir = args.cache_dir
    device = args.device

    log_directory = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file_path = os.path.join(log_directory, "preprocess.log")
    logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

    logger = logging.getLogger(__name__)
    
    main()
