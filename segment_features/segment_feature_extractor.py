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
    return parser.parse_args()


# Video Processing
class VideoProcessor:
    def __init__(self, method, feature_extractor, video_transform):
        self.method = method
        self.feature_extractor = feature_extractor
        self.video_transform = video_transform

        self.fps = 30
        self.num_frames_per_feature = 30

    def process_video(self, video_name, video_directory_path, output_features_path):
        segment_size = self.fps / self.num_frames_per_feature
        video_path = os.path.join(video_directory_path, f"{video_name}.mp4" if "mp4" not in video_name else video_name)

        output_file_path = os.path.join(output_features_path, video_name)

        if os.path.exists(f"{output_file_path}_{int(segment_size)}s_{int(1)}s.npz"):
            logger.info(f"Skipping video: {video_name}")
            return

        os.makedirs(output_features_path, exist_ok=True)

        video = EncodedVideo.from_path(video_path)
        video_duration = video.duration - 0.0

        logger.info(f"video: {video_name} video_duration: {video_duration} s")
        segment_end = max(video_duration - segment_size + 1, 1)
        stride = 1

        video_features = []
        for start_time in tqdm(np.arange(0, segment_end, segment_size),
                               desc=f"Processing video segments for video {video_name}"):
            end_time = start_time + segment_size
            end_time = min(end_time, video_duration)

            if end_time - start_time < 0.04:
                continue

            video_data = video.get_clip(start_sec=start_time, end_sec=end_time)
            segment_video_inputs = video_data["video"]

            segment_features = extract_features(
                video_data_raw=segment_video_inputs,
                feature_extractor=self.feature_extractor,
                transforms_to_apply=self.video_transform,
                method=self.method
            )

            video_features.append(segment_features)

        video_features = np.vstack(video_features)
        np.savez(f"{output_file_path}_{int(segment_size)}s_{int(stride)}s.npz", video_features)
        logger.info(f"Finished extraction and saving video: {video_name} video_features: {video_features.shape}")


# Feature Extraction
def extract_features(video_data_raw, feature_extractor, transforms_to_apply, method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_data_for_transform = {"video": video_data_raw, "audio": None}
    video_data = transforms_to_apply(video_data_for_transform)
    video_inputs = video_data["video"]
    
    if method in ["omnivore"]:
        video_input = video_inputs[0][None, ...].to(device)
    elif method == "slowfast":
        video_input = [i.to(device)[None, ...] for i in video_inputs]
    elif method == "x3d":
        video_input = video_inputs.unsqueeze(0).to(device)
    elif method == "3dresnet":
        video_input = video_inputs.unsqueeze(0).to(device)
    elif method == "egovlp":
        video_input = video_inputs.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [B, C, T, H, W]
    elif method == "perception_encoder":
        # PE expects [T, C, H, W] format (time, channels, height, width)
        # PyTorchVideo returns [C, T, H, W] so we need to permute
        video_input = video_inputs.permute(1, 0, 2, 3).to(device)  # [C, T, H, W] -> [T, C, H, W]
    with torch.no_grad():
        if method == "perception_encoder":
            # Process frames and average pool temporal dimension
            features = feature_extractor(video_input)  # [T, D]
            features = features.mean(dim=0, keepdim=True)  # [1, D] - average over time
        else:
            features = feature_extractor(video_input)
    return features.cpu().numpy()


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
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        egovlp_model = FrozenInTime(
            video_params={
                "model": "SpaceTimeTransformer",
                "arch_config": "base_patch16_224",
                "num_frames": 16,
                "pretrained": False,  # Don't load ImageNet pretrained weights, we'll load egovlp.pth
                "time_init": "zeros"
            },
            text_params={
                "model": "distilbert-base-uncased",
                "pretrained": True,
                "input": "text"
            },
            projection="minimal",
            load_checkpoint="skip",  # Non-empty value to skip ViT checkpoint loading
        )
        egovlp_model.load_state_dict(checkpoint['state_dict'], strict=False)
        egovlp_model.eval()
        
        # Create wrapper to handle EgoVLP's expected input format
        class EgoVLPWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, video_tensor):
                # EgoVLP expects dict with 'video' key and video_only=True
                # Ensure video_only is passed as a keyword argument
                data = {'video': video_tensor}
                return self.model.forward(data, video_only=True)
        
        model = EgoVLPWrapper(egovlp_model)
    elif name == "perception_encoder":
        # Add perception_models to path
        pe_path = os.path.join(os.path.dirname(__file__), '..', 'lib', 'perception_models')
        sys.path.insert(0, pe_path)
        
        import core.vision_encoder.pe as pe
        
        # Load PE-Core-L14-336 model (good balance of performance and size)
        # Available: PE-Core-T16-384, PE-Core-S16-384, PE-Core-B16-224, PE-Core-L14-336, PE-Core-G14-448
        model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
        # Extract only the visual encoder for feature extraction
        model = model.visual

    feature_extractor = model
    feature_extractor = feature_extractor.to(device)
    feature_extractor = feature_extractor.eval()
    return feature_extractor


def main_hololens(is_sequential=False):
    hololens_directory_path = "/data/rohith/captain_cook/data/hololens/"
    output_features_path = f"/data/rohith/captain_cook/features/hololens/segments/{method}/"

    video_transform = get_video_transformation(method)
    feature_extractor = get_feature_extractor(method)

    processor = VideoProcessor(method, feature_extractor, video_transform)

    if not is_sequential:
        num_threads = 10
        with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
            for recording_id in os.listdir(hololens_directory_path):
                video_file_path = os.path.join(hololens_directory_path, recording_id, "sync", "pv")
                executor.submit(processor.process_video, recording_id, video_file_path, output_features_path)
    else:
        for recording_id in os.listdir(hololens_directory_path):
            video_file_path = os.path.join(hololens_directory_path, recording_id, "sync", "pv")
            processor.process_video(recording_id, video_file_path, output_features_path)


# Main
def main():
    video_files_path = "../data/video/"
    output_features_path = f"../data/features/gopro/segments/{method}/"

    video_transform = get_video_transformation(method)
    feature_extractor = get_feature_extractor(method)

    processor = VideoProcessor(method, feature_extractor, video_transform)

    mp4_files = [file for file in os.listdir(video_files_path) if file.endswith(".mp4")]
    
    # Limit number of videos if specified
    if max_videos is not None:
        mp4_files = mp4_files[:max_videos]
    
    logger.info(f"Processing {len(mp4_files)} videos")

    num_threads = 1
    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
        list(
            tqdm(
                executor.map(
                    lambda file: processor.process_video(file, video_files_path, output_features_path), mp4_files
                ), total=len(mp4_files)
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for processing methods.")
    parser.add_argument("--backbone", type=str, default="omnivore", help="Specify the method to be used.")
    args = parse_arguments()
    method = args.backbone
    max_videos = args.max_videos

    log_directory = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file_path = os.path.join(log_directory, f"{method}.log")
    logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

    logger = logging.getLogger(__name__)

    # main_hololens(is_sequential=False)
    main()
