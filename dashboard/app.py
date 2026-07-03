#!/usr/bin/env python3
"""
TrapTracker dashboard — a local, branded launcher + status page for the
DeepFaune-UK pipeline. Reads the dataset folders directly (no database) and
surfaces current dataset/model versions, pool size, and links to the tools.

Local use only — no authentication. Do not expose to the internet.
"""
import os, json, glob
from flask import Flask, render_template, jsonify, abort, Response

app = Flask(__name__)
DATASET_ROOT = os.environ.get('DATASET_ROOT', '/dataset')
TUTORIAL_PATH = os.environ.get('TUTORIAL_PATH', '/app/TUTORIAL.md')

# cache the computed status for a few seconds so scanning a large pool doesn't
# run on every single request (the page and the 30s auto-refresh reuse it).
import time as _time
_CACHE = {'at': 0.0, 'data': None}
_CACHE_TTL = float(os.environ.get('STATUS_CACHE_TTL', '15'))

# The tool URLs are the host-published ports (the dashboard links the browser to
# them; they're not proxied through Flask).
JUPYTER_URL = os.environ.get('JUPYTER_URL', 'http://localhost:8888/lab')
LABELSTUDIO_URL = os.environ.get('LABELSTUDIO_URL', 'http://localhost:8080')
# base for deep-linking to a specific notebook, e.g. {JUPYTER_BASE}/lab/tree/01_ingest_autolabel.ipynb
JUPYTER_BASE = os.environ.get('JUPYTER_BASE', 'http://localhost:8888')


def _human(n):
    for u in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f'{n:.0f} {u}'
        n /= 1024
    return f'{n:.0f} PB'


def gather_status():
    """Read live pipeline state from the dataset directory."""
    s = {
        'pool_images': 0, 'pool_size': '0 B',
        'dataset_versions': [], 'latest_dataset': None,
        'models': [], 'latest_model': None, 'latest_f1': None,
        'incoming': 0, 'awaiting_verification': False,
        'classes': 0,
    }

    # pool — count only (fast). We deliberately do NOT sum file sizes here:
    # os.path.getsize on tens of thousands of files over a bind-mount makes the
    # page hang. Count is one cheap listdir.
    img_dir = os.path.join(DATASET_ROOT, 'pool', 'images')
    if os.path.isdir(img_dir):
        try:
            with os.scandir(img_dir) as it:
                s['pool_images'] = sum(1 for e in it if e.is_file())
        except OSError:
            s['pool_images'] = 0
    s['pool_size'] = None  # not computed (kept for template compatibility)

    # dataset manifests
    mdir = os.path.join(DATASET_ROOT, 'manifests')
    if os.path.isdir(mdir):
        vers = sorted(f[len('dataset-'):-len('.json')] for f in os.listdir(mdir)
                      if f.startswith('dataset-') and f.endswith('.json'))
        s['dataset_versions'] = vers
        if vers:
            s['latest_dataset'] = vers[-1]
            try:
                man = json.load(open(os.path.join(mdir, f'dataset-{vers[-1]}.json')))
                s['classes'] = len(man.get('classes', []))
                s['splits'] = man.get('split_counts', {})
            except Exception:
                pass

    # models
    models_dir = os.path.join(DATASET_ROOT, 'models')
    if os.path.isdir(models_dir):
        for v in sorted(os.listdir(models_dir)):
            card_path = os.path.join(models_dir, v, 'model_card.json')
            onnx = glob.glob(os.path.join(models_dir, v, '*.onnx'))
            entry = {'version': v, 'has_onnx': bool(onnx), 'f1': None}
            if os.path.exists(card_path):
                try:
                    card = json.load(open(card_path))
                    entry['f1'] = card.get('macro_f1_test')
                except Exception:
                    pass
            s['models'].append(entry)
        if s['models']:
            latest = s['models'][-1]
            s['latest_model'] = latest['version']
            s['latest_f1'] = latest['f1']

    # incoming
    inc = os.path.join(DATASET_ROOT, 'incoming')
    if os.path.isdir(inc):
        s['incoming'] = len([f for f in os.listdir(inc)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    # awaiting verification = ingest produced tasks but no export yet
    has_tasks = os.path.exists(os.path.join(DATASET_ROOT, 'ls_import', 'tasks.json'))
    has_export = os.path.exists(os.path.join(DATASET_ROOT, 'ls_export', 'export.json'))
    s['awaiting_verification'] = has_tasks and not has_export

    return s


def cached_status():
    """gather_status() with a short TTL cache so large-pool scans don't run on
    every request."""
    now = _time.time()
    if _CACHE['data'] is None or (now - _CACHE['at']) > _CACHE_TTL:
        _CACHE['data'] = gather_status()
        _CACHE['at'] = now
    return _CACHE['data']


def _find_logo():
    """Return the filename of a logo in static/ if present, else None.
    Drop a file named logo.png / logo.svg / logo.jpg into dashboard/static/."""
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    for name in ('logo.png', 'logo.svg', 'logo.jpg', 'logo.jpeg', 'logo.webp'):
        if os.path.exists(os.path.join(static_dir, name)):
            return name
    return None


@app.route('/')
def index():
    js = JUPYTER_BASE.rstrip('/')
    def nb(name):
        return f'{js}/lab/tree/{name}'
    stages = [
        {'num': '00',  'name': 'Setup',      'desc': 'Check the environment, GPU, and that your models are found.',       'url': nb('00_setup.ipynb'),          'kind': 'notebook'},
        {'num': '00b', 'name': 'Bootstrap',  'desc': 'Seed the pool from your existing dataset (one-time).',              'url': nb('00b_bootstrap.ipynb'),     'kind': 'notebook'},
        {'num': '01',  'name': 'Ingest',     'desc': 'Auto-label new images in incoming/ with the current model.',        'url': nb('01_ingest_autolabel.ipynb'),'kind': 'notebook'},
        {'num': '02',  'name': 'Verify',     'desc': 'Confirm or correct detections in Label Studio.',                    'url': LABELSTUDIO_URL,               'kind': 'labelstudio'},
        {'num': '03',  'name': 'Merge',      'desc': 'Add verified images to the append-only pool; new manifest.',        'url': nb('03_merge.ipynb'),          'kind': 'notebook'},
        {'num': '04',  'name': 'Retrain',    'desc': 'Rebuild the classifier from the versioned manifest (cached crops).','url': nb('04_retrain.ipynb'),        'kind': 'notebook'},
        {'num': '05',  'name': 'Evaluate',   'desc': 'Score on held-out test, export ONNX, write the model card.',        'url': nb('05_evaluate.ipynb'),       'kind': 'notebook'},
        {'num': '06',  'name': 'Inference',  'desc': 'Run the latest model on new / out-of-distribution images.',         'url': nb('06_inference.ipynb'),      'kind': 'notebook'},
        {'num': '07',  'name': 'Cleanup',    'desc': 'Reset temporary state for the next iteration (safe).',              'url': nb('07_cleanup.ipynb'),        'kind': 'notebook'},
    ]
    return render_template('index.html',
                           status=cached_status(),
                           logo=_find_logo(),
                           jupyter_url=JUPYTER_URL,
                           labelstudio_url=LABELSTUDIO_URL,
                           stages=stages)


@app.route('/api/status')
def api_status():
    return jsonify(cached_status())


def _md_to_html(md):
    """Minimal, dependency-free markdown -> HTML (headings, code, lists, tables,
    bold/inline-code, paragraphs). Enough for the tutorial."""
    import re, html as _html
    lines = md.split('\n')
    out = []
    i = 0
    in_code = False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('```'):
            if not in_code:
                out.append('<pre><code>'); in_code = True
            else:
                out.append('</code></pre>'); in_code = False
            i += 1; continue
        if in_code:
            out.append(_html.escape(ln)); i += 1; continue
        # tables
        if '|' in ln and i+1 < len(lines) and set(lines[i+1].replace('|','').strip()) <= set('-: '):
            header = [c.strip() for c in ln.strip().strip('|').split('|')]
            out.append('<table><thead><tr>' + ''.join(f'<th>{_html.escape(c)}</th>' for c in header) + '</tr></thead><tbody>')
            i += 2
            while i < len(lines) and '|' in lines[i]:
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                out.append('<tr>' + ''.join(f'<td>{_inline(c)}</td>' for c in cells) + '</tr>')
                i += 1
            out.append('</tbody></table>'); continue
        m = re.match(r'(#{1,4})\s+(.*)', ln)
        if m:
            lvl = len(m.group(1)); out.append(f'<h{lvl}>{_inline(m.group(2))}</h{lvl}>'); i += 1; continue
        if re.match(r'\s*[-*]\s+', ln):
            out.append('<ul>')
            while i < len(lines) and re.match(r'\s*[-*]\s+', lines[i]):
                out.append('<li>' + _inline(re.sub(r'\s*[-*]\s+','',lines[i],count=1)) + '</li>'); i += 1
            out.append('</ul>'); continue
        if re.match(r'\s*\d+\.\s+', ln):
            out.append('<ol>')
            while i < len(lines) and re.match(r'\s*\d+\.\s+', lines[i]):
                out.append('<li>' + _inline(re.sub(r'\s*\d+\.\s+','',lines[i],count=1)) + '</li>'); i += 1
            out.append('</ol>'); continue
        if ln.strip() == '---':
            out.append('<hr>'); i += 1; continue
        if ln.strip() == '':
            i += 1; continue
        out.append('<p>' + _inline(ln) + '</p>'); i += 1
    return '\n'.join(out)


def _inline(s):
    import re, html as _html
    s = _html.escape(s)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
    return s


@app.route('/tutorial')
def tutorial():
    if not os.path.exists(TUTORIAL_PATH):
        abort(404)
    body = _md_to_html(open(TUTORIAL_PATH, encoding='utf-8').read())
    return render_template('tutorial.html', body=body)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
