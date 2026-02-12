"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import copy
import torch
from video_llama.common.registry import registry
from decord import VideoReader
import decord
import numpy as np
from video_llama.processors import transforms_video
from video_llama.processors.base_processor import BaseProcessor
from video_llama.processors.randaugment import VideoRandomAugment
from video_llama.processors import functional_video as F
from omegaconf import OmegaConf
from torchvision import transforms
import random as rnd
import librosa
import warnings
warnings.filterwarnings("ignore")

MAX_INT = registry.get("MAX_INT")
decord.bridge.set_bridge("torch")

def find_frames(video_file, num_frames=8):
    y, sr = librosa.load(video_file, sr=44100)

    # Define a function to find the peak in each segment
    def find_peak_in_segment(feature, start, end):
        segment = feature[:, start:end]
        if segment.size == 0:  # Ensure the segment is not empty
            return None
        peak_index = np.argmax(np.sum(segment, axis=0))  # Sum across MFCC coefficients or other features
        return start + peak_index  # Return the index in the full feature

    # Compute Spectral Flux
    S = np.abs(librosa.stft(y, window='hamming'))
    spectral_flux = np.sum(np.diff(S, axis=1)**2, axis=0)

    # Get the number of time frames for the feature (e.g., MFCC)
    num_spectral_flux_frames = len(spectral_flux)

    # Divide into 8 segments based on the number of time frames for MFCC
    num_segments = num_frames
    # segment_length_mfcc = num_mfcc_frames // num_segments
    segment_length_flux = num_spectral_flux_frames // num_segments

    # Initialize an empty list to store selected frame indices
    selected_peaks = []

    # Loop through each segment and find the most significant peak
    for i in range(num_segments):
        # Calculate segment start and end for each feature
        start_flux = i * segment_length_flux
        end_flux = (i + 1) * segment_length_flux
        
        # Find peaks in each feature segment
        peak_flux = find_peak_in_segment(np.expand_dims(spectral_flux, axis=0), start_flux, end_flux)
        
        # Store the peak (you can choose which one to prioritize)
        selected_peaks.append(peak_flux)  # Example: Using Spectral Flux

    # Convert selected peaks (time indices) into time frames (if needed for video frames)
    frame_times = librosa.frames_to_time(selected_peaks, sr=sr)

    return frame_times

def load_video(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", return_msg = False):

    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_path, height=height, width=width)

    sampling = "uniform"
    
    vlen = len(vr)
    start, end = 0, vlen

    if vlen<=n_frms:
        indices = np.arange(start, end).astype(int).tolist()
        extra_needed = n_frms - vlen
        extra_indices = rnd.choices(indices, k=extra_needed)
        indices += extra_indices
        indices.sort()
    else:
        if sampling == "uniform":
            indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
        elif sampling == "headtail":
            indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
            indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
            indices = indices_h + indices_t
        else:
            raise NotImplementedError

    if vlen > n_frms:
        frame_times = find_frames(video_path, n_frms)
        indices = [int(ft* vr.get_avg_fps()) for ft in frame_times]

        def adjust_list(input_list, vlen):
            new_list = copy.copy(input_list)
            if input_list[0] < 0:
                new_list[0] = 0
            for i in range(1, len(input_list)):
                if input_list[i] == input_list[i - 1]:
                    new_list[i] = input_list[i] + 1
                if new_list[i] >= vlen:
                    new_list[i] = vlen-1
            return new_list
        
        indices = adjust_list(indices, vlen)

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(indices)
    # print(type(temp_frms))
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    fps = float(vr.get_avg_fps())
    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    # " " should be added in the start and end
    msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    return frms, msg


class AlproVideoBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None, n_frms=MAX_INT):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms_video.NormalizeVideo(mean, std)

        self.n_frms = n_frms


class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__


class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__


class ResizeVideo(object):
    def __init__(self, target_size, interpolation_mode="bilinear"):
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return F.resize(clip, self.target_size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + "(resize_size={0})".format(self.target_size)


@registry.register_processor("alpro_video_train")
class AlproVideoTrainProcessor(AlproVideoBaseProcessor):
    def __init__(
        self,
        image_size=384,
        sampling = "uniform",
        mean=None,
        std=None,
        min_scale=0.5,
        max_scale=1.0,
        n_frms=MAX_INT,
    ):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size
        self.sampling = sampling

        self.transform = transforms.Compose(
            [
                # Video size is (C, T, H, W)
                transforms_video.RandomResizedCropVideo(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation_mode="bicubic",
                ),
                ToTHWC(),  # C, T, H, W -> T, H, W, C
                ToUint8(),
                transforms_video.ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                self.normalize,
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        clip = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
            sampling="headtail",
        )

        return self.transform(clip)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        n_frms = cfg.get("n_frms", MAX_INT)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            n_frms=n_frms,
        )


@registry.register_processor("alpro_video_eval")
class AlproVideoEvalProcessor(AlproVideoBaseProcessor):
    def __init__(self, image_size=256, mean=None, std=None, n_frms=MAX_INT):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size

        # Input video size is (C, T, H, W)
        self.transform = transforms.Compose(
            [
                # frames will be resized during decord loading.
                ToUint8(),  # C, T, H, W
                ToTHWC(),  # T, H, W, C
                transforms_video.ToTensorVideo(),  # C, T, H, W
                self.normalize,  # C, T, H, W
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        clip = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
        )

        return self.transform(clip)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        n_frms = cfg.get("n_frms", MAX_INT)

        return cls(image_size=image_size, mean=mean, std=std, n_frms=n_frms)
