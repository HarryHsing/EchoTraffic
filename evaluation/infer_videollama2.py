"""Zero-shot Description inference with the separately installed VideoLLaMA2 API."""
import argparse
import gc
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.common import annotation_key, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--video-root', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--videollama2-source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0, help='0: all 3000 Description samples; positive values: diagnostic subset')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(Path(args.videollama2_source).resolve()))
    import numpy as np
    import torch
    from videollama2 import model_init, mm_infer
    import videollama2.mm_utils as mm_utils
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def seed(n):
        random.seed(n)
        np.random.seed(n)
        torch.manual_seed(n)
        torch.cuda.manual_seed_all(n)

    seed(42)
    rows = json.loads(Path(args.annotations).read_text())
    samples = []
    for index, row in enumerate(rows):
        key = annotation_key(row)
        if key.endswith('--discription'):
            video = Path(args.video_root)/key.rsplit('--', 1)[0]
            if not video.is_file():
                raise FileNotFoundError(video)
            samples.append((index, key, str(video), row['QA'][0]['q']))
    del rows
    if len(samples) != 3000:
        raise ValueError(f'Expected full 3000 Description annotations, got {len(samples)}')
    if args.limit:
        samples = samples[:args.limit]
    decoded, audio_calls, video_calls = [], [], []
    decode = mm_utils.load_audio_from_video

    def recorded_decode(*a, **kw):
        value = decode(*a, **kw)
        decoded.append(True)
        return value

    mm_utils.load_audio_from_video = recorded_decode
    model, processor, tokenizer = model_init(args.model_path, device_map={'': 'cuda:0'})
    model.eval()
    if model.config.num_frames != 8 or model.config.model_type != 'videollama2_qwen2':
        raise ValueError('Use the pinned VideoLLaMA2.1-7B-AV checkpoint')
    tower = model.get_audio_tower()
    extract = tower.extract_features

    def recorded_extract(*a, **kw):
        value = extract(*a, **kw)
        audio_calls.append(True)
        return value

    tower.extract_features = recorded_extract
    hook = model.get_vision_tower().register_forward_hook(lambda *unused: video_calls.append(True))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.metadata.json').write_text(json.dumps({
        'n': len(samples), 'diagnostic_subset': bool(args.limit),
        'annotation_sha256': sha256(args.annotations), 'frames': 8,
        'dtype': str(next(model.parameters()).dtype), 'do_sample': False,
        'max_new_tokens': 2048, 'chat_template': tokenizer.chat_template,
        'finetuned_on_AV_TAU': False}, indent=2))
    with output.open('x') as handle, torch.inference_mode():
        for done, (index, key, video, question) in enumerate(samples, 1):
            seed(42 + index)
            decoded.clear()
            audio_calls.clear()
            video_calls.clear()
            tensor = processor['video'](video, va=True)
            if not decoded or not isinstance(tensor, dict) or not {'audio','video'} <= set(tensor):
                raise RuntimeError(f'Audio/video decoding failed: {key}')
            if not all(bool(torch.isfinite(v).all()) for v in tensor.values()):
                raise RuntimeError(f'Non-finite audio/video input: {key}')
            prediction = mm_infer(tensor, question, model=model, tokenizer=tokenizer,
                                  modal='video', do_sample=False, max_new_tokens=2048)
            if not audio_calls or not video_calls:
                raise RuntimeError(f'Audio or video branch did not execute: {key}')
            handle.write(json.dumps({'key': key, 'qa_index': index, 'prediction': prediction}, ensure_ascii=False)+'\n')
            handle.flush()
            print(json.dumps({'done': done, 'total': len(samples)}), flush=True)
            del tensor
            gc.collect()
            torch.cuda.empty_cache()
    hook.remove()


if __name__ == '__main__':
    main()
