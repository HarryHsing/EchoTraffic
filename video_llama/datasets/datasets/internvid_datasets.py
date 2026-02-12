"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.datasets.datasets.caption_datasets import CaptionDataset
import pandas as pd
import decord
from decord import VideoReader
import random
import torch
import json
import ffmpeg
from torch.utils.data.dataloader import default_collate
from video_llama.models.ImageBind.data import load_and_transform_audio_data
class InternvidDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_root):
        """
        vis_root (string): Root directory of video (e.g. webvid_eval/video/)
        ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)


        # 读取一个路径下所有的
        anno_data = []
        cnt = 0 
        with open(ann_root, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                anno_data.append(data)
                cnt += 1

        self.annotation = anno_data
        self.vis_root = vis_root
        self.resize_size = 224
        self.num_frm = 8
        # self.frm_sampling_strategy = 'headtail'

    def collater(self, samples):
        return {
            'image': torch.stack([x['image'] for x in samples]),
            'audio': torch.stack([x['audio'] for x in samples]),
            'text_input': [x['text_input'] for x in samples],
            'type': [x['type'] for x in samples]
        }
        # return default_collate(samples)


    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                sample_dict = self.annotation[index]

                if 'Caption' in sample_dict.keys():
                    text = sample_dict['Caption']
                    caption = self.text_processor(text)
                else:
                    raise NotImplementedError("Un-supported text annotation format.")

                # fetch video
                video_lst = [i for i in os.listdir(self.vis_root) if 'InternVId' in i]
                for v in video_lst:
                    video_path = self.vis_root+v+'/{}_{}_{}.mp4'.format(sample_dict['YoutubeID'], sample_dict['Start_timestamp'], sample_dict['End_timestamp'])
                    if os.path.exists(video_path): 
                        break
                # video_path = self.vis_root+'/{}_{}_{}.mp4'.format(sample_dict['YoutubeID'], sample_dict['Start_timestamp'], sample_dict['End_timestamp'])
                video = self.vis_processor(video_path)
                
                # audio_path = video_path.replace('.mp4', '.wav')
                audio = load_and_transform_audio_data([video_path], video.device,  clips_per_video=8)[0]
                # if not os.path.exists(audio_path): 
                #     audio = None
                # else:
                #     audio = load_and_transform_audio_data([audio_path], video.device,  clips_per_video=8)[0]
            except:
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        return {
            "image": video,
            "audio": audio,
            "text_input": caption,
            "type":'video',
        }

    def __len__(self):
        return len(self.annotation)

    # def collater(self, samples):
    #     new_result = {}
    #     new_result['image'] = default_collate( [sample["image"] for sample in samples])
    #     new_result['text_input'] = default_collate( [sample["text_input"] for sample in samples])
    #     return new_result
