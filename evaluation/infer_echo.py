"""Batch audio-video inference; reference answers never enter generation."""
import argparse
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.common import TEXT_TASKS, annotation_key, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--video-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tasks', nargs='+', choices=TEXT_TASKS, default=list(TEXT_TASKS))
    parser.add_argument('--cfg-path', default='eval_configs/finetune_eval.yaml')
    parser.add_argument('--model-path', default='ckpt/videollama_video_audio_sft/checkpoint_0.pth')
    parser.add_argument('--options', nargs='+', default=None)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-per-task', type=int, default=0, help='0 scores the full task; positive values are diagnostic only')
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    annotations = json.loads(Path(args.annotations).read_text())
    counts = dict.fromkeys(args.tasks, 0)
    samples = []
    for index, row in enumerate(annotations):
        key = annotation_key(row)
        task = key.rsplit('--', 1)[1]
        if task not in counts or (args.limit_per_task and counts[task] >= args.limit_per_task):
            continue
        video = Path(args.video_root) / key.rsplit('--', 1)[0]
        if not video.is_file():
            raise FileNotFoundError(video)
        # Retain only the question and identifiers. No reference in the model loop.
        samples.append((index, key, str(video), row['QA'][0]['q']))
        counts[task] += 1
    if not samples or len({s[1] for s in samples}) != len(samples):
        raise ValueError('Empty selection or duplicate video/task keys')
    if not args.limit_per_task and any(n != 3000 for n in counts.values()):
        raise ValueError(f'Full test split requires 3000 samples per selected task: {counts}')
    del annotations
    import torch
    from inference import VideoLLaMaInference
    torch.set_num_threads(4)
    engine = VideoLLaMaInference(SimpleNamespace(
        cfg_path=args.cfg_path, model_path=args.model_path, options=args.options,
        gpu_id=args.gpu_id, seed=args.seed, num_frames=8, model_type='llama2'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'counts': counts, 'diagnostic_subset': bool(args.limit_per_task),
                'annotation_sha256': sha256(args.annotations), 'sft_sha256': sha256(args.model_path),
                'seed_rule': f'{args.seed} + index in full annotation list',
                'frames': 8, 'loading_audit': engine.loading_audit,
                'do_sample': False, 'num_beams': 1, 'max_new_tokens': 500, 'max_length': 2000}
    destination.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2))
    with destination.open('x') as output, torch.inference_mode():
        for done, (index, key, video, question) in enumerate(samples, 1):
            prediction = engine.process_video(video, question, seed=args.seed + index)
            if prediction is None:
                raise RuntimeError(f'Inference failed for {key}; partial output must not be scored as a full run')
            output.write(json.dumps({'key': key, 'qa_index': index, 'prediction': prediction}, ensure_ascii=False) + '\n')
            output.flush()
            print(json.dumps({'done': done, 'total': len(samples)}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
