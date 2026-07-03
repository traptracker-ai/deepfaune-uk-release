"""
inference_lib.py — two-stage ONNX inference (YOLO detector + DeepFaune-UK classifier).

This is the proven pipeline from development, packaged for reuse by the staged
notebooks. Pure onnxruntime + numpy + pillow; no Ultralytics at runtime.
"""

import json
import numpy as np
from PIL import Image

from dataset_lib import CLASSES as CLF_CLASSES  # 30-class classifier order

# Detector's 31-class map (includes Car at 26; Person at 7).
YOLO_NAMES = ['ErinaceusEuropaeus','SciurusCarolinensis','SciurusVulgaris','CapreolusCapreolus',
              'CervusElaphus','VulpesVulpes','MelesMeles','Person','CapraHircus','OryctolagusCuniculus',
              'ColumbaPalumbus','PhasianusColchicus','OvisAries','PasserDomesticus','DamaDama','BosTaurus',
              'FelisCatus','MartesMartes','CanisFamiliaris','CalibrationPole','AccipiterGentilis','ButeoButeo',
              'CappercaillieCock','CappercaillieHen','NumeniusArquata','NumeniusArquataChick','Car',
              'EquusCaballus','CorvusCorone','SusScrofa','MuntiacusReevesi']


# ---- classifier preprocessing ----
def _resize_pad_square(img, size):
    w, h = img.size
    s = size / max(w, h)
    nw, nh = round(w*s), round(h*s)
    img = img.resize((nw, nh), Image.BILINEAR)
    c = Image.new('RGB', (size, size), (0,0,0))
    c.paste(img, ((size-nw)//2, (size-nh)//2))
    return c

def _prep_clf(crop, size, mean, std):
    a = np.asarray(_resize_pad_square(crop.convert('RGB'), size), np.float32)/255.0
    a = (a - np.array(mean, np.float32)) / np.array(std, np.float32)
    return a.transpose(2,0,1)[None,...].astype(np.float32)

def _softmax(x):
    e = np.exp(x - x.max()); return e/e.sum()

def load_classifier_meta(onnx_path):
    import onnx
    m = onnx.load(onnx_path)
    meta = {p.key: p.value for p in m.metadata_props}
    return (json.loads(meta['classes']), int(meta.get('input_size', 518)),
            json.loads(meta.get('mean', '[0.485,0.456,0.406]')),
            json.loads(meta.get('std', '[0.229,0.224,0.225]')))


# ---- detector pre/post ----
def _letterbox(img, new=640, color=(114,114,114)):
    w, h = img.size
    r = min(new/w, new/h)
    nw, nh = int(round(w*r)), int(round(h*r))
    c = Image.new('RGB', (new, new), color)
    px, py = (new-nw)//2, (new-nh)//2
    c.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    a = np.asarray(c, np.float32)/255.0
    return a.transpose(2,0,1)[None,...].astype(np.float32), r, px, py

def _nms(boxes, scores, iou_thr):
    if len(boxes) == 0: return []
    x1,y1,x2,y2 = boxes.T
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]; keep=[]
    while order.size>0:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        w=np.maximum(0,xx2-xx1); h=np.maximum(0,yy2-yy1); inter=w*h
        ovr=inter/(areas[i]+areas[order[1:]]-inter+1e-9)
        order=order[1:][ovr<=iou_thr]
    return keep

def _unlb(x1,y1,x2,y2,r,px,py,W,H):
    x1=(x1-px)/r; y1=(y1-py)/r; x2=(x2-px)/r; y2=(y2-py)/r
    return (max(0,min(W,x1)),max(0,min(H,y1)),max(0,min(W,x2)),max(0,min(H,y2)))

def _decode(out, nclz, conf, iou, r, px, py, W, H):
    if out.ndim==3: out=out[0]
    if out.ndim==2 and out.shape[1]==6:  # end-to-end
        return [(*_unlb(*d[:4],r,px,py,W,H), float(d[4]), int(d[5]))
                for d in out if d[4]>=conf]
    ch=4+nclz
    if out.shape[0]==ch: out=out.T
    elif out.shape[1]!=ch:
        raise ValueError(f'bad YOLO output {out.shape}, want channels={ch}')
    cls=out[:,4:4+nclz]; cid=cls.argmax(1); cf=cls.max(1)
    m=cf>=conf; box=out[m,:4]; cf=cf[m]; cid=cid[m]
    if len(cf)==0: return []
    cx,cy,bw,bh=box.T
    xy=np.stack([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2],1)
    keep=_nms(xy,cf,iou)
    return [(*_unlb(*xy[i],r,px,py,W,H), float(cf[i]), int(cid[i])) for i in keep]


class Pipeline:
    """Loads both ONNX models once; call .infer(image_path) per image."""
    def __init__(self, detector_onnx, classifier_onnx, providers=None):
        import onnxruntime as ort
        providers = providers or ['CUDAExecutionProvider','CPUExecutionProvider']
        self.det = ort.InferenceSession(detector_onnx, providers=providers)
        self.clf = ort.InferenceSession(classifier_onnx, providers=providers)
        self.det_in = self.det.get_inputs()[0].name
        self.clf_in = self.clf.get_inputs()[0].name
        self.classes, self.size, self.mean, self.std = load_classifier_meta(classifier_onnx)

    def infer(self, image_path, det_imgsz=640, det_conf=0.25, nms_iou=0.45, pad=0.12):
        """Returns list of detections dicts:
           {box:(x1,y1,x2,y2), detector_class, detector_conf, species, species_conf, kind}
           kind ∈ {'animal','person','vehicle'}."""
        frame = Image.open(image_path).convert('RGB')
        W, H = frame.size
        inp, r, px, py = _letterbox(frame, det_imgsz)
        out = self.det.run(None, {self.det_in: inp})[0]
        dets = _decode(out, len(YOLO_NAMES), det_conf, nms_iou, r, px, py, W, H)
        results = []
        for (x1,y1,x2,y2,dconf,cid) in dets:
            dname = YOLO_NAMES[cid] if 0<=cid<len(YOLO_NAMES) else str(cid)
            cl = dname.lower()
            rec = {'box': (x1,y1,x2,y2), 'detector_class': dname,
                   'detector_conf': round(dconf,4)}
            if cl in ('person','human'):
                rec.update(kind='person', species='Person', species_conf=round(dconf,4))
            elif cl in ('vehicle','car'):
                rec.update(kind='vehicle', species=dname, species_conf=None)
            else:
                bw,bh=x2-x1,y2-y1
                cx1=max(0,int(x1-bw*pad)); cy1=max(0,int(y1-bh*pad))
                cx2=min(W,int(x2+bw*pad)); cy2=min(H,int(y2+bh*pad))
                crop = frame.crop((cx1,cy1,cx2,cy2))
                logits = self.clf.run(None, {self.clf_in: _prep_clf(crop,self.size,self.mean,self.std)})[0][0]
                probs = _softmax(logits); top=int(probs.argmax())
                rec.update(kind='animal', species=self.classes[top],
                           species_conf=round(float(probs[top]),4))
            results.append(rec)
        return frame, results
