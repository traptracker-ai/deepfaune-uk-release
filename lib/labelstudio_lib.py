"""
labelstudio_lib.py — bridge between the inference pipeline and Label Studio.

Two directions:
  * to_ls_tasks(...)  : turn pipeline predictions into LS import tasks WITH
    pre-annotations, so the reviewer corrects rather than labels from scratch.
  * ls_export_to_yolo(...) : parse LS's exported JSON (after human review) back
    into YOLO-format label lines for merging into the pool.

Label Studio's rectangle values use PERCENT coordinates (x, y, width, height as
% of image size, with x,y the TOP-LEFT corner). We convert to/from YOLO
(normalised cx, cy, w, h) here.
"""

import json, os
from PIL import Image
from dataset_lib import CLASSES, NAME_TO_ID


def _xyxy_to_ls_rect(x1, y1, x2, y2, W, H):
    """Pixel xyxy → Label Studio percent rect (x,y top-left, width,height)."""
    return {
        'x': 100.0 * x1 / W,
        'y': 100.0 * y1 / H,
        'width': 100.0 * (x2 - x1) / W,
        'height': 100.0 * (y2 - y1) / H,
    }


def to_ls_task(image_ls_path, image_size, detections, from_name='label',
               to_name='image', min_conf_for_preannot=0.0):
    """Build one Label Studio task dict with pre-annotations.
    image_ls_path: the path LS uses to serve the image (e.g. /data/local-files/?d=incoming/x.jpg)
    detections: list from Pipeline.infer() (species + box per detection)."""
    W, H = image_size
    results = []
    for d in detections:
        if d['species_conf'] is not None and d['species_conf'] < min_conf_for_preannot:
            # still show the box, but it'll be flagged low-confidence; keep it
            pass
        x1, y1, x2, y2 = d['box']
        rect = _xyxy_to_ls_rect(x1, y1, x2, y2, W, H)
        label = d['species']  # Person/animal species/vehicle name
        results.append({
            'from_name': from_name, 'to_name': to_name, 'type': 'rectanglelabels',
            'value': {**rect, 'rectanglelabels': [label]},
            'score': d.get('species_conf') or d.get('detector_conf') or 0.0,
        })
    return {
        'data': {'image': image_ls_path},
        'predictions': [{'model_version': 'auto-label-v1', 'result': results}],
    }


def build_label_config(classes=None):
    """Label Studio labeling config XML for box+label review over our classes."""
    classes = classes or (CLASSES + ['Car', 'Vehicle'])
    labels = '\n'.join(f'    <Label value="{c}"/>' for c in classes)
    return f"""<View>
  <Image name="image" value="$image" zoom="true"/>
  <RectangleLabels name="label" toName="image">
{labels}
  </RectangleLabels>
</View>"""


def ls_export_to_yolo(export_json_path, images_dir, out_labels_dir, id_from='filename'):
    """Parse a Label Studio JSON-export (after review) into YOLO label files.

    For each task, reads the verified rectangle annotations and writes
    <stem>.txt with `class_id cx cy w h` lines (normalised). Returns a dict
    {image_filename: n_boxes}.

    id_from='filename' keys output by the image filename stem (so it lines up
    with the pool image ids)."""
    os.makedirs(out_labels_dir, exist_ok=True)
    data = json.load(open(export_json_path))
    written = {}
    for task in data:
        # locate the image reference
        img_ref = task.get('data', {}).get('image', '')
        stem = _stem_from_ref(img_ref)
        if not stem:
            continue
        # find the *verified* annotation (annotations[0].result), not predictions
        anns = task.get('annotations') or []
        if not anns:
            continue
        result = anns[0].get('result', [])
        lines = []
        for r in result:
            if r.get('type') != 'rectanglelabels':
                continue
            v = r['value']
            labels = v.get('rectanglelabels', [])
            if not labels:
                continue
            name = labels[0]
            if name not in NAME_TO_ID:
                # skip classes not in the trainable set (e.g. Vehicle/Car)
                continue
            cid = NAME_TO_ID[name]
            # LS percent (top-left x,y,width,height) → YOLO normalised cx,cy,w,h
            cx = (v['x'] + v['width']/2) / 100.0
            cy = (v['y'] + v['height']/2) / 100.0
            w  = v['width'] / 100.0
            h  = v['height'] / 100.0
            lines.append(f'{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
        out_path = os.path.join(out_labels_dir, stem + '.txt')
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        written[stem] = len(lines)
    return written


def _stem_from_ref(ref):
    """Extract the image filename stem from an LS data.image reference, which may
    look like '/data/local-files/?d=incoming/abc.jpg' or a plain path."""
    if not ref:
        return None
    # take the part after the last '/' or '=', then drop extension
    tail = ref.replace('\\', '/').split('?d=')[-1].split('/')[-1]
    return os.path.splitext(tail)[0] or None
