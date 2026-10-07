import argparse
import os
import random
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import decord
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, default_conversation, conv_llava_llama_2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoLLaMaInference:
    def __init__(self, args):
        # Reset GPU memory statistics
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        self.args = args
        self.cfg = Config(args)
        self.setup_environment()
        self.model = self.load_model(args.model_path)
        self.vis_processor = self.setup_processor()
        self.chat = Chat(self.model, self.vis_processor, device=f'cuda:{args.gpu_id}')

    def get_peak_memory(self) -> float:
        """Get current GPU peak memory usage in MB"""
        return torch.cuda.max_memory_allocated() / 1024 / 1024

    def setup_environment(self):
        """Set up environment and random seeds"""
        self.set_seed(self.args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        decord.bridge.set_bridge('torch')

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def load_model(self, model_path: str):
        """Load and initialize the model"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if checkpoint.get('config', {}).get('model', {}).get('equip_audio_branch') is not True:
                raise ValueError('Use the corrected audio-video SFT checkpoint; the previous release has no SFT audio weights.')
            state = checkpoint['model']
            if not any(key.startswith('audio_') for key in state):
                raise ValueError('The SFT checkpoint does not contain the audio branch.')
            model_config = self.cfg.model_cfg
            model_config.device_8bit = self.args.gpu_id
            model_cls = registry.get_model_class(model_config.arch)
            model = model_cls.from_config(model_config)
            result = model.load_state_dict(state, strict=False)
            missing = sorted(name for name, p in model.named_parameters()
                             if p.requires_grad and name not in state)
            if missing or result.unexpected_keys:
                raise RuntimeError(f'Incomplete SFT load: missing trainable={missing}; unexpected={result.unexpected_keys}')
            self.loading_audit = {'sft_state_entries': len(state),
                                  'missing_trainable': missing,
                                  'unexpected_keys': list(result.unexpected_keys)}
            model = model.to(f'cuda:{self.args.gpu_id}')
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def setup_processor(self):
        """Set up vision processor"""
        vis_processor_cfg = self.cfg.datasets_cfg.my_dataset_instruct.vis_processor.train
        return registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    def process_video(self, video_path: str, prompt: str, seed=None) -> Optional[str]:
        """Process video and generate response"""
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            if seed is not None:
                self.set_seed(seed)

            chat_state = conv_llava_llama_2.copy() if self.args.model_type == 'llama2' else default_conversation.copy()
            img_list = []

            # Upload video
            audio_calls = []
            hook = self.model.audio_Qformer.bert.register_forward_hook(
                lambda *unused: audio_calls.append(True))
            try:
                self.chat.upload_video(video_path, chat_state, img_list,
                                       num_frames=self.args.num_frames)
            finally:
                hook.remove()
            if not audio_calls:
                raise RuntimeError('Audio processing failed; refusing a silent video-only fallback.')

            # Ask question and get response
            self.chat.ask(prompt, chat_state)
            self.last_prompt = chat_state.get_prompt()
            response = self.chat.answer(
                conv=chat_state,
                img_list=img_list,
                num_beams=1,
                temperature=1.0,
                max_new_tokens=500,
                max_length=2000,
                do_sample=False
            )[0]

            return response
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Video-LLaMA Inference")
    parser.add_argument("--cfg-path", 
                       default='./eval_configs/finetune_eval.yaml',
                       help="path to configuration file.")
    parser.add_argument("--model-path", 
                       default='./ckpt/videollama_video_audio_sft/checkpoint_0.pth',
                       help="path to model.")
    parser.add_argument("--gpu-id", type=int, default=0, help="Specify the GPU to load the model.")
    parser.add_argument("--model_type", type=str, default='llama2', help="The type of LLM")
    parser.add_argument("--video-path", 
                       type=str, 
                       default="./test_video/036680.mp4",
                       help="Path to the input video.")
    parser.add_argument("--prompt", 
                       type=str, 
                       default="What unusual event takes place in the video?",
                       help="Prompt content for the model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames used in the verified audio-video inference setup")
    parser.add_argument("--options", default=["run.seed=42"], nargs="+", 
                       help="Override some settings in the used config.") 
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    inference = VideoLLaMaInference(args)
    response = inference.process_video(args.video_path, args.prompt)
    if response is None:
        raise RuntimeError('Video inference failed; see the error above.')
    

    logger.info("Processing Results:")
    logger.info(f"Video Path: {args.video_path}")
    logger.info(f"Question: {args.prompt}")
    logger.info(f"Model Response: {response}")
    logger.info(f"Peak GPU Memory Usage: {inference.get_peak_memory():.2f} MB")

if __name__ == "__main__":
    main()
