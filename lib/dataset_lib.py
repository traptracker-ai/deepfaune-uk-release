"""
dataset_lib.py — core dataset management for the DeepFaune-UK pipeline.

Design principles:
  * The image/label POOL is append-only. Images are stored once, keyed by a
    content hash, and never moved or deleted.
  * Train/val/test split is assigned DETERMINISTICALLY from the image hash, so a
    given image always lands in the same split across every rebuild — no leakage
    of a crop drifting between train and test.
  * Each dataset VERSION is a lightweight manifest (JSON) listing which image IDs
    belong to it and their split. To reproduce a version, replay its manifest.
"""

import os, json, hashlib, shutil, time
from collections import defaultdict

# Canonical 30-class list (Car dropped). Order is authoritative for training.
CLASSES = ['ErinaceusEuropaeus','SciurusCarolinensis','SciurusVulgaris','CapreolusCapreolus',
           'CervusElaphus','VulpesVulpes','MelesMeles','Person','CapraHircus','OryctolagusCuniculus',
           'ColumbaPalumbus','PhasianusColchicus','OvisAries','PasserDomesticus','DamaDama','BosTaurus',
           'FelisCatus','MartesMartes','CanisFamiliaris','CalibrationPole','AccipiterGentilis','ButeoButeo',
           'CappercaillieCock','CappercaillieHen','NumeniusArquata','NumeniusArquataChick',
           'EquusCaballus','CorvusCorone','SusScrofa','MuntiacusReevesi']
NAME_TO_ID = {n: i for i, n in enumerate(CLASSES)}


def image_id(path):
    """Content hash of an image file → stable unique id (first 16 hex chars)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def assign_split(img_id, val_frac=0.10, test_frac=0.10):
    """Deterministic split from the image id. Same id → same split forever.
    Uses a second hash of the id mapped to [0,1)."""
    hv = int(hashlib.sha256(('split:' + img_id).encode()).hexdigest(), 16)
    frac = (hv % 10_000_000) / 10_000_000.0
    if frac < test_frac:
        return 'test'
    if frac < test_frac + val_frac:
        return 'val'
    return 'train'


def pool_paths(dataset_root):
    return (os.path.join(dataset_root, 'pool', 'images'),
            os.path.join(dataset_root, 'pool', 'labels'))


def add_to_pool(dataset_root, src_image, yolo_label_lines):
    """Copy an image into the pool under its content-hash id and write its YOLO
    label file. Idempotent: if the id already exists, it is not duplicated.
    Returns (img_id, split, was_new)."""
    img_dir, lbl_dir = pool_paths(dataset_root)
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)
    iid = image_id(src_image)
    ext = os.path.splitext(src_image)[1].lower()
    dst_img = os.path.join(img_dir, iid + ext)
    dst_lbl = os.path.join(lbl_dir, iid + '.txt')
    was_new = not os.path.exists(dst_img)
    if was_new:
        shutil.copy2(src_image, dst_img)
    # labels are (re)written — verification may have corrected them
    with open(dst_lbl, 'w') as f:
        f.write('\n'.join(yolo_label_lines) + ('\n' if yolo_label_lines else ''))
    return iid, assign_split(iid), was_new


def find_pool_image(dataset_root, iid):
    img_dir, _ = pool_paths(dataset_root)
    for f in os.listdir(img_dir):
        if f.startswith(iid + '.'):
            return os.path.join(img_dir, f)
    return None


def list_pool(dataset_root):
    """All image ids currently in the pool that have a (non-empty) label file."""
    img_dir, lbl_dir = pool_paths(dataset_root)
    if not os.path.isdir(img_dir):
        return []
    ids = []
    for f in sorted(os.listdir(img_dir)):
        iid = os.path.splitext(f)[0]
        lbl = os.path.join(lbl_dir, iid + '.txt')
        if os.path.exists(lbl):
            ids.append(iid)
    return ids


def write_manifest(dataset_root, version, note=''):
    """Snapshot the current pool into a versioned manifest. Records every image
    id, its split, and the class counts — the reproducible definition of a
    dataset version."""
    img_dir, lbl_dir = pool_paths(dataset_root)
    ids = list_pool(dataset_root)
    entries, class_counts, split_counts = [], defaultdict(int), defaultdict(int)
    for iid in ids:
        split = assign_split(iid)
        split_counts[split] += 1
        # tally classes from the label file
        lbl = os.path.join(lbl_dir, iid + '.txt')
        cls_in_img = []
        for line in open(lbl):
            line = line.split()
            if line:
                cid = int(float(line[0]))
                class_counts[cid] += 1
                cls_in_img.append(cid)
        entries.append({'id': iid, 'split': split, 'n_boxes': len(cls_in_img)})

    manifest = {
        'version': version,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'note': note,
        'classes': CLASSES,
        'n_images': len(ids),
        'split_counts': dict(split_counts),
        'class_counts': {CLASSES[k]: v for k, v in sorted(class_counts.items())},
        'images': entries,
    }
    mdir = os.path.join(dataset_root, 'manifests')
    os.makedirs(mdir, exist_ok=True)
    path = os.path.join(mdir, f'dataset-{version}.json')
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
    return path, manifest


def load_manifest(dataset_root, version):
    path = os.path.join(dataset_root, 'manifests', f'dataset-{version}.json')
    with open(path) as f:
        return json.load(f)


def latest_manifest_version(dataset_root):
    mdir = os.path.join(dataset_root, 'manifests')
    if not os.path.isdir(mdir):
        return None
    vers = [f[len('dataset-'):-len('.json')] for f in os.listdir(mdir)
            if f.startswith('dataset-') and f.endswith('.json')]
    return sorted(vers)[-1] if vers else None
