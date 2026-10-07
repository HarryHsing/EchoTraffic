"""Shared identifiers for the released AV-TAU annotation format."""
import hashlib
from pathlib import Path

TEXT_TASKS = ('discription', 'reason', 'prevention', 'response')


def normalize_key(key):
    video, task = key.rsplit('--', 1)
    path = Path(video)
    # Historical paths use <category>/<id>.mp4; the release flattens this
    # to <category>_<id>.mp4. Keep the release's 'discription' spelling.
    video_id = path.parent.name + '_' + path.name if path.is_absolute() else path.name
    task = {'description': 'discription', 'causation': 'reason'}.get(task, task)
    return video_id + '--' + task


def annotation_key(row):
    return normalize_key(row['video'] + '--' + row['type'])


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
