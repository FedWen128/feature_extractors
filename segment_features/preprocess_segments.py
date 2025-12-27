import argparse
import os
import sys
import json

# Add the parent directory to the Python path so we can import omnivore_transforms
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
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
    return parser.parse_args()


# Video Preprocessing
class SegmentPreprocessor:
    def __init__(self, cache_dir, num_threads=10):
        self.cache_dir = cache_dir
        self.num_threads = num_threads
        self.fps = 30
        self.num_frames_per_feature = 30
        
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
            video = EncodedVideo.from_path(video_path)
            video_duration = video.duration - 0.0
        except Exception as e:
            logger.error(f"Failed to load video {video_name}: {e}")
            return

        logger.info(f"video: {video_name} video_duration: {video_duration} s")
        segment_end = max(video_duration - segment_size + 1, 1)
        
        segments = []
        segment_index = 0
        
        for start_time in tqdm(np.arange(0, segment_end, segment_size),
                               desc=f"Preprocessing video segments for {video_name}"):
            end_time = start_time + segment_size
            end_time = min(end_time, video_duration)

            if end_time - start_time < 0.04:
                continue

            try:
                video_data = video.get_clip(start_sec=start_time, end_sec=end_time)
                segment_video = video_data["video"].numpy()  # Convert to numpy for storage
                
                # Save segment
                segment_path = os.path.join(video_cache_dir, f"segment_{segment_index:06d}.npy")
                np.save(segment_path, segment_video)
                
                segments.append({
                    "index": segment_index,
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "path": segment_path
                })
                segment_index += 1
            except Exception as e:
                logger.error(f"Failed to process segment at {start_time}s for {video_name}: {e}")
                continue

        # Save metadata
        metadata = {
            "video_name": video_name,
            "video_duration": video_duration,
            "segment_size": segment_size,
            "total_segments": len(segments),
            "segments": segments
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Finished preprocessing video: {video_name} with {len(segments)} segments")


def main():
    video_files_path = "../data/video/"
    
    preprocessor = SegmentPreprocessor(cache_dir=cache_dir, num_threads=num_threads)

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

    log_directory = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file_path = os.path.join(log_directory, "preprocess.log")
    logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

    logger = logging.getLogger(__name__)
    
    main()
