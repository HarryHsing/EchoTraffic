"""The four traditional Table 2 metrics with explicit reference/prediction order."""
import argparse
import collections
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.common import TEXT_TASKS, annotation_key, normalize_key, sha256


def load_pairs(annotations, predictions, tasks, allow_subset):
    rows = json.loads(Path(annotations).read_text())
    references = {}
    for row in rows:
        key = annotation_key(row)
        if key.rsplit('--', 1)[1] not in tasks:
            continue
        if key in references:
            raise ValueError(f'Duplicate reference: {key}')
        references[key] = row['QA'][0]['a']
    candidates = {}
    with open(predictions) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = normalize_key(row['key'])
            if key.rsplit('--', 1)[1] not in tasks:
                continue
            if 'error' in row or not isinstance(row.get('prediction'), str):
                raise ValueError(f'Failed or invalid prediction: {key}')
            if key in candidates:
                raise ValueError(f'Duplicate prediction: {key}')
            if key not in references:
                raise ValueError(f'Unknown prediction key: {key}')
            candidates[key] = row['prediction']
    if not candidates:
        raise ValueError('No predictions selected')
    if not allow_subset:
        if set(candidates) != set(references):
            raise ValueError(f'Incomplete coverage: {len(candidates)} predictions / {len(references)} references')
        for task in tasks:
            n = sum(k.endswith('--' + task) for k in candidates)
            if n != 3000:
                raise ValueError(f'{task}: expected 3000 pairs, got {n}')
    # Preserve prediction-file order, including each inference task's batch order.
    return [(key, references[key], pred) for key, pred in candidates.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tasks', nargs='+', choices=TEXT_TASKS, default=list(TEXT_TASKS))
    parser.add_argument('--allow-subset', action='store_true', help='Diagnostic only; never label a subset as a Table 2 result')
    parser.add_argument('--metric-model-root', required=True,
                        help='Directory with distilbert-base-uncased/ and roberta-large/')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    pairs = load_pairs(args.annotations, args.predictions, args.tasks, args.allow_subset)
    metadata = {'annotation_sha256': sha256(args.annotations), 'prediction_sha256': sha256(args.predictions),
                'n': len(pairs), 'diagnostic_subset': args.allow_subset,
                'mover_source_sha256': sha256(Path(__file__).parent/'vendor/moverscore_v2.py')}
    model_root = Path(args.metric_model_root).resolve()
    if not (model_root/'distilbert-base-uncased').is_dir() or not (model_root/'roberta-large').is_dir():
        raise FileNotFoundError('Download both pinned metric backbones listed in evaluation/metric_models.json')
    import torch
    import jieba
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from bert_score import BERTScorer
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        raise RuntimeError('The retained MoverScore implementation requires CUDA')
    # Keep the exact model name: the retained source branches on this string.
    # Replacing it with an absolute path would select a different hidden-state tuple.
    vendor = Path(__file__).resolve().parent/'vendor/moverscore_v2.py'
    os.chdir(model_root)
    os.environ.pop('MOVERSCORE_MODEL', None)
    spec = importlib.util.spec_from_file_location('echotraffic_moverscore', vendor)
    mover = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mover)
    bert = BERTScorer(model_type=str(model_root/'roberta-large'), num_layers=17,
                      idf=False, rescale_with_baseline=False, device='cuda',
                      use_fast_tokenizer=False, batch_size=16)
    bert._tokenizer.model_max_length = 512
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method7
    tokenize = lambda text: ' '.join(jieba.cut(text)).split()
    groups = collections.defaultdict(list)
    # Keep tasks separate so padding/batching matches the verified audit.
    for task in args.tasks:
        task_pairs = [pair for pair in pairs if pair[0].endswith('--' + task)]
        for start in range(0, len(task_pairs), 16):
            batch = task_pairs[start:start+16]
            refs = [p[1] for p in batch]
            preds = [p[2] for p in batch]
            _, _, bs = bert.score(preds, refs, batch_size=16)
            ms = mover.word_mover_score(refs, preds, collections.defaultdict(lambda: 1.),
                                       collections.defaultdict(lambda: 1.), stop_words=[],
                                       n_gram=1, remove_subwords=False, batch_size=16)
            for (_, ref, pred), b, m in zip(batch, bs, ms):
                groups[task].append({
                    'bleu4': sentence_bleu([tokenize(ref)], tokenize(pred), weights=(.25,)*4, smoothing_function=smooth),
                    'rouge_p': rouge.score(ref, pred)['rougeL'].precision,
                    'moverscore': float(m), 'bertscore': float(b)})
            if start % 256 == 0:
                print(json.dumps({'task': task, 'scored': min(start+16, len(task_pairs)), 'total': len(task_pairs)}), flush=True)
    def summarize(values):
        return {'n': len(values), 'mean': {k: statistics.mean(v[k] for v in values) for k in values[0]}}
    scores = {task: summarize(values) for task, values in groups.items()}
    if set(groups) == set(TEXT_TASKS):
        scores['overall_four_text_tasks'] = summarize([v for group in groups.values() for v in group])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump({'metadata': metadata, 'scores': scores}, handle, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
