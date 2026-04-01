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
        seed = (self.cfg.run_cfg.seed or 0) + get_rank()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        decord.bridge.set_bridge('torch')

    def load_model(self, model_path: str):
        """Load and initialize the model"""
        try:
            checkpoint = torch.load(model_path)
            model_config = self.cfg.model_cfg
            model_config.device_8bit = self.args.gpu_id
            model_cls = registry.get_model_class(model_config.arch)
            model = model_cls.from_config(model_config).to(f'cuda:{self.args.gpu_id}')
            model.load_state_dict(checkpoint['model'], strict=False)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def setup_processor(self):
        """Set up vision processor"""
        vis_processor_cfg = self.cfg.datasets_cfg.my_dataset_instruct.vis_processor.train
        return registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    def process_video(self, video_path: str, prompt: str) -> Optional[str]:
        """Process video and generate response"""
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            chat_state = conv_llava_llama_2.copy() if self.args.model_type == 'llama2' else default_conversation.copy()
            img_list = []

            # Upload video
            self.chat.upload_video(video_path, chat_state, img_list, num_frames=self.args.num_frames)

            # Ask question and get response
            self.chat.ask(prompt, chat_state)
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
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    "CUDA out of memory during video upload/inference. "
                    "Root cause is usually high frame count (num_frames=%s), which increases visual token and embedding memory. "
                    "Try a smaller value such as --num-frames 8.",
                    self.args.num_frames,
                )
            else:
                logger.error(f"Runtime error processing video: {str(e)}")
            return None
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
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of video frames to sample for inference. Larger values use more GPU memory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--options", default=["run.seed=42"], nargs="+", 
                       help="Override some settings in the used config.") 
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    inference = VideoLLaMaInference(args)
    response = inference.process_video(args.video_path, args.prompt)
    

    logger.info("Processing Results:")
    logger.info(f"Video Path: {args.video_path}")
    logger.info(f"Question: {args.prompt}")
    logger.info(f"Model Response: {response}")
    logger.info(f"Peak GPU Memory Usage: {inference.get_peak_memory():.2f} MB")

if __name__ == "__main__":
    main()
