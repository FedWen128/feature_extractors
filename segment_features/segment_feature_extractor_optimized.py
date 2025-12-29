import argparse
import datetime
import glob
import os
import sys

# Add the parent directory to the Python path so we can import omnivore_transforms
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
import torchvision.transforms as T
import concurrent.futures
import logging
from tqdm import tqdm
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo
from omnivore_transforms import SpatialCrop, TemporalCrop
from PIL import Image
from natsort import natsorted


# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for processing methods.")
    parser.add_argument("--backbone", type=str, default="omnivore", help="Specify the method to be used.")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of videos to process")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    return parser.parse_args()


# Video Processing
class VideoProcessor:
    def __init__(self, method, feature_extractor, video_transform, batch_size=8):
        self.method = method
        self.feature_extractor = feature_extractor
        self.video_transform = video_transform
        self.batch_size = batch_size

        self.fps = 30
        self.num_frames_per_feature = 30
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process_video(self, video_name, video_directory_path, output_features_path):
        segment_size = self.fps / self.num_frames_per_feature
        video_path = os.path.join(video_directory_path, f"{video_name}.mp4" if "mp4" not in video_name else video_name)

        output_file_path = os.path.join(output_features_path, video_name)

        if os.path.exists(f"{output_file_path}_{int(segment_size)}s_{int(1)}s.npz"):
            logger.info(f"Skipping video: {video_name}")
            return

        os.makedirs(output_features_path, exist_ok=True)

        try:
            video = EncodedVideo.from_path(video_path)
        except Exception as e:
            logger.error(f"Failed to load video {video_name}: {e}")
            return

        video_duration = video.duration - 0.0

        logger.info(f"video: {video_name} video_duration: {video_duration} s")
        segment_end = max(video_duration - segment_size + 1, 1)
        stride = 1

        video_features = []
        batch_inputs = []

        # Collect segments
        for start_time in tqdm(np.arange(0, segment_end, segment_size),
                               desc=f"Processing video segments for video {video_name}"):
            end_time = start_time + segment_size
            end_time = min(end_time, video_duration)

            if end_time - start_time < 0.04:
                continue

            video_data = video.get_clip(start_sec=start_time, end_sec=end_time)
            segment_video_inputs = video_data["video"]

            # Apply transforms immediately to save memory and prepare for batching
            video_data_for_transform = {"video": segment_video_inputs, "audio": None}
            video_data_transformed = self.video_transform(video_data_for_transform)
            video_input = video_data_transformed["video"]
            
            batch_inputs.append(video_input)

            if len(batch_inputs) >= self.batch_size:
                self._process_batch(batch_inputs, video_features)
                batch_inputs = []

        # Process remaining items
        if batch_inputs:
            self._process_batch(batch_inputs, video_features)

        if not video_features:
            logger.warning(f"No features extracted for {video_name}")
            return

        video_features = np.vstack(video_features)
        np.savez(f"{output_file_path}_{int(segment_size)}s_{int(stride)}s.npz", video_features)
        logger.info(f"Finished extraction and saving video: {video_name} video_features: {video_features.shape}")

    def _process_batch(self, batch_inputs, video_features):
        # Stack inputs based on method requirements
        if self.method == "omnivore":
            # Omnivore expects [B, C, T, H, W]
            # video_input is [C, T, H, W]
            batch_tensor = torch.stack(batch_inputs).to(self.device)
        elif self.method == "slowfast":
            # SlowFast expects list of [B, C, T, H, W]
            # video_input is list of [C, T, H, W] (slow and fast pathways)
            # We need to stack slow pathways and fast pathways separately
            slow_pathways = torch.stack([b[0] for b in batch_inputs]).to(self.device)
            fast_pathways = torch.stack([b[1] for b in batch_inputs]).to(self.device)
            batch_tensor = [slow_pathways, fast_pathways]
        elif self.method in ["x3d", "3dresnet"]:
             # Expects [B, C, T, H, W]
             batch_tensor = torch.stack(batch_inputs).to(self.device)
        elif self.method == "egovlp":
            # EgoVLP expects [B, T, C, H, W] (based on my analysis of SpaceTimeTransformer)
            # video_input is [C, T, H, W]
            # permute to [T, C, H, W] then stack -> [B, T, C, H, W]
            batch_tensor = torch.stack([v.permute(1, 0, 2, 3) for v in batch_inputs]).to(self.device)
        elif self.method == "perception_encoder":
            # PE expects [T, C, H, W] for single input, but for batch?
            # Usually CLIP-like models expect [B, T, C, H, W] or [B*T, C, H, W] depending on implementation
            # The original code did: video_input = video_inputs.permute(1, 0, 2, 3).to(device) # [T, C, H, W]
            # features = feature_extractor(video_input) # [T, D]
            # features = features.mean(dim=0, keepdim=True) # [1, D]
            
            # If we want to batch, we probably need to process each video in the batch separately 
            # or reshape if the model supports batching over videos.
            # PE-Core (CLIP visual) typically takes [B, C, H, W].
            # Here we have temporal dimension.
            # We can stack all frames from all videos: [B*T, C, H, W]
            # Then reshape back to [B, T, D] and mean pool.
            
            # video_input is [C, T, H, W]
            # permute to [T, C, H, W]
            processed_inputs = [v.permute(1, 0, 2, 3) for v in batch_inputs]
            # Stack: [B, T, C, H, W]
            batch_tensor = torch.stack(processed_inputs).to(self.device)
            
            # Reshape for CLIP: [B*T, C, H, W]
            b, t, c, h, w = batch_tensor.shape
            batch_tensor = batch_tensor.view(b * t, c, h, w)
            
        with torch.no_grad():
            # Use mixed precision for speed
            with torch.amp.autocast('cuda'):
                if self.method == "perception_encoder":
                    features = self.feature_extractor(batch_tensor) # [B*T, D]
                    # Reshape back to [B, T, D]
                    features = features.view(b, t, -1)
                    # Mean pool over time
                    features = features.mean(dim=1) # [B, D]
                elif self.method == "egovlp":
                     # EgoVLPWrapper expects dict with 'video'
                     # And the wrapper I saw in previous file handles the dict creation
                     # But here I am calling feature_extractor directly which is the wrapper
                     features = self.feature_extractor(batch_tensor)
                else:
                    features = self.feature_extractor(batch_tensor)
        
        video_features.append(features.cpu().numpy())


# Model Initialization
def get_video_transformation(name):
    if name == "omnivore":
        num_frames = 32
        video_transform = T.Compose(
            [
                UniformTemporalSubsample(num_frames),
                T.Lambda(lambda x: x / 255.0),
                ShortSideScale(size=224),
                NormalizeVideo(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                TemporalCrop(frames_per_clip=32, stride=40),
                SpatialCrop(crop_size=224, num_crops=3),
            ]
        )
    elif name == "slowfast":
        slowfast_alpha = 4
        num_frames = 32
        side_size = 256
        crop_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]

        class PackPathway(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, frames: torch.Tensor):
                fast_pathway = frames
                # Perform temporal sampling from the fast pathway.
                slow_pathway = torch.index_select(
                    frames,
                    1,
                    torch.linspace(
                        0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
                    ).long(),
                )
                frame_list = [slow_pathway, fast_pathway]
                return frame_list

        video_transform = T.Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size),
                PackPathway(),
            ]
        )
    elif name == "x3d":
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        model_transform_params = {
            "x3d_xs": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 4,
                "sampling_rate": 12,
            },
            "x3d_s": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 13,
                "sampling_rate": 6,
            },
            "x3d_m": {
                "side_size": 256,
                "crop_size": 256,
                "num_frames": 16,
                "sampling_rate": 5,
            },
        }
        # Taking x3d_m as the model
        transform_params = model_transform_params["x3d_m"]
        video_transform = Compose(
            [
                UniformTemporalSubsample(transform_params["num_frames"]),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=transform_params["side_size"]),
                CenterCropVideo(
                    crop_size=(
                        transform_params["crop_size"],
                        transform_params["crop_size"],
                    )
                ),
            ]
        )
    elif name == "3dresnet":
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 8
        video_transform = Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size=(crop_size, crop_size)),
            ]
        )
    elif name == "egovlp":
        num_frames = 16
        video_transform = T.Compose([
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            ShortSideScale(size=224),
            CenterCropVideo(crop_size=224),
            NormalizeVideo(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    elif name == "perception_encoder":
        # PE-Core uses standard CLIP normalization
        num_frames = 16  # Good balance for video understanding
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        video_transform = Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=336),  # PE-Core-L14-336
                CenterCropVideo(crop_size=336),
            ]
        )
    return ApplyTransformToKey(key="video", transform=video_transform)


def get_feature_extractor(name, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    if name == "omnivore":
        model_name = "omnivore_swinB_epic"
        model = torch.hub.load("facebookresearch/omnivore:main", model=model_name)
        model.heads = torch.nn.Identity()
    elif name == "slowfast":
        model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
        model.heads = torch.nn.Identity()
    elif name == "x3d":
        model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=True)
        model.heads = torch.nn.Identity()
    elif name == "3dresnet":
        model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
        model.heads = torch.nn.Identity()
    elif name == "egovlp":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib', 'EgoVLP'))
        from model.model import FrozenInTime
        
        # Load config and checkpoint
        checkpoint_path = "../lib/EgoVLP/pretrained/egovlp.pth"
        if not os.path.exists(checkpoint_path):
             # Fallback or error
             print(f"Warning: Checkpoint not found at {checkpoint_path}")

        # We need to handle the case where checkpoint might not exist or we want to download it
        # For now assuming it exists as per original code
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except:
            checkpoint = {'state_dict': {}}

        egovlp_model = FrozenInTime(
            video_params={
                "model": "SpaceTimeTransformer",
                "arch_config": "base_patch16_224",
                "num_frames": 16,
                "pretrained": False,
                "time_init": "zeros"
            },
            text_params={
                "model": "distilbert-base-uncased",
                "pretrained": True,
                "input": "text"
            },
            projection="minimal",
            load_checkpoint="skip",
        )
        if 'state_dict' in checkpoint:
            egovlp_model.load_state_dict(checkpoint['state_dict'], strict=False)
        egovlp_model.eval()
        
        class EgoVLPWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, video_tensor):
                data = {'video': video_tensor}
                return self.model.forward(data, video_only=True)
        
        model = EgoVLPWrapper(egovlp_model)
    
    elif name == "perception_encoder":
        pe_path = os.path.join(os.path.dirname(__file__), '..', 'lib', 'perception_models')
        sys.path.insert(0, pe_path)
        
        import core.vision_encoder.pe as pe
        
        model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
        model = model.visual

    feature_extractor = model
    feature_extractor.to(device)
    feature_extractor.eval()

    # Compilation
    try:
        print("Compiling model with torch.compile...")
        feature_extractor = torch.compile(feature_extractor)
    except Exception as e:
        print(f"Compilation failed or not supported: {e}")

    return feature_extractor


# Main
def main():
    args = parse_arguments()
    method = args.backbone
    max_videos = args.max_videos
    batch_size = args.batch_size

    video_files_path = "/content/drive/MyDrive/AMLproject/input_videos/"
    output_features_path = f"/content/drive/MyDrive/AMLproject/our_features/gopro/segments/{method}/"

    # Setup logging
    log_directory = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file_path = os.path.join(log_directory, f"{method}_optimized.log")
    logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
    
    global logger
    logger = logging.getLogger(__name__)

    video_transform = get_video_transformation(method)
    feature_extractor = get_feature_extractor(method)

    processor = VideoProcessor(method, feature_extractor, video_transform, batch_size=batch_size)

    if not os.path.exists(video_files_path):
        print(f"Video path {video_files_path} does not exist.")
        return

    mp4_files = [file for file in os.listdir(video_files_path) if file.endswith(".mp4")]
    
    if max_videos is not None:
        mp4_files = mp4_files[:max_videos]
    
    logger.info(f"Processing {len(mp4_files)} videos with batch size {batch_size}")

    # Sequential processing for simplicity with batching inside process_video
    # Or parallel processing of videos (but GPU memory might be an issue if batching is also used)
    # Let's stick to sequential video processing but batched frames for now to be safe on memory
    for file in tqdm(mp4_files, desc="Processing videos"):
        processor.process_video(file, video_files_path, output_features_path)


if __name__ == "__main__":
    main()
