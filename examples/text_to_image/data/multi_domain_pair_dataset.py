                              
import os, glob, math, random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

                                          
IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp"]

def _stem(p: str) -> str:
                                  
    return os.path.splitext(os.path.basename(p))[0]

def _read_img(path: str, transform) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transform(img)

def _read_mask(path: str, transform) -> torch.Tensor:

    m = Image.open(path).convert("L")       

                                                                
    if hasattr(transform, "transforms"):
        t_list = []
        for t in transform.transforms:
            if isinstance(t, T.Normalize):
                continue              
            t_list.append(t)
        mask_tf = T.Compose(t_list)
    else:
        mask_tf = transform 

    m_tensor = mask_tf(m)                                      

                 
    if m_tensor.ndim == 2:
        m_tensor = m_tensor.unsqueeze(0)
    elif m_tensor.ndim == 3 and m_tensor.shape[0] == 3:
                                    
        m_tensor = m_tensor[:1, ...]

                
    if m_tensor.max() > 1.0:
        m_tensor = m_tensor / 255.0
    m_tensor = m_tensor.clamp(0.0, 1.0).to(dtype=torch.float32)
    return m_tensor

def _np_load(path: str) -> np.ndarray:
    a = np.load(path, allow_pickle=False)
    if a.ndim >= 1 and a.shape[0] == 1:
        a = a.squeeze(0)
    return a

def _load_tensor_pt_or_npy(path_pt: str, path_npy: str) -> torch.Tensor:
    if os.path.exists(path_pt):
        t = torch.load(path_pt, map_location="cpu")
        return t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
    if os.path.exists(path_npy):
        return torch.tensor(_np_load(path_npy), dtype=torch.float32)
    raise FileNotFoundError(f"feature not found: {path_pt} or {path_npy}")

def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    na = (a * a).sum() ** 0.5
    nb = (b * b).sum() ** 0.5
    if na < eps or nb < eps: 
        return 0.0
    return float((a @ b) / (na * nb))

                                         
@dataclass
class DomainSpec:
    name: str
    image_dir: str                          
    arcface_dir: str                                                          
    whole_arcface_dir: str                                               
    clip_dir: str                                              
    lmk_dir: Optional[str] = None                                    
    mask_dir: Optional[str] = None                  
    img_exts: List[str] = None             
    train_count: int = 0
    test_count: int = 0
    group: str = ""                                                  

    def __post_init__(self):
        if self.img_exts is None:
            self.img_exts = IMG_EXTS

                                                    
class MultiDomainPairDataset(Dataset):

    def __init__(
        self,
        domains: List[DomainSpec],
        split: Literal["train", "test"],
        transform,
        mix_weights: Dict[str, float],
        cross_domain_p: float = 0.6,
        mode: Literal["inpaint", "swap"] = "swap",
        self_pair_prob: float = 0.0,
        avoid_same_identity: bool = False,
        same_id_cos_thr: float = 0.8,
        seed: int = 42,
    ):
        assert split in ["train", "test"]
        self.domains = domains
        self.split = split
        self.tf = transform
        self.cross_domain_p = float(max(0.0, min(1.0, cross_domain_p)))
        self.mode = mode
        self.self_pair_prob = float(max(0.0, min(1.0, self_pair_prob)))
        self.avoid_same_identity = avoid_same_identity
        self.same_id_cos_thr = float(same_id_cos_thr)

                                                   
        self.base_seed = int(seed)
        self.epoch = 0
        self.rng = random.Random(self.base_seed)

                                                 
                                                          
        self.domain_items: Dict[str, List[str]] = {}
        for d in self.domains:
            paths = []
            for ext in d.img_exts:
                                
                paths += glob.glob(os.path.join(d.image_dir, "**", f"*{ext}"), recursive=True)

            def _numkey(p):
                s = _stem(p)
                try:
                    return int(s)                  
                except:
                    return s
            paths = sorted(paths, key=_numkey)

            need_train = d.train_count
            need_test  = d.test_count
            if need_train + need_test > 0:
                assert len(paths) >= need_train + need_test,\
                    f"[{d.name}] images {len(paths)} < train({need_train}) + test({need_test})"

            if self.split == "train":
                use = paths[:need_train] if need_train > 0 else paths
            else:
                use = paths[-need_test:] if need_test > 0 else []
            self.domain_items[d.name] = use

                                                
        self.effective_w: Dict[str, float] = {}
        group_members: Dict[str, List[str]] = {}
        for d in self.domains:
            if d.group:
                group_members.setdefault(d.group, []).append(d.name)

        for g, members in group_members.items():
            if g in mix_weights:
                w_each = mix_weights[g] / max(1, len(members))
                for m in members:
                    self.effective_w[m] = self.effective_w.get(m, 0.0) + w_each

        for d in self.domains:
            if d.name in mix_weights:
                self.effective_w[d.name] = self.effective_w.get(d.name, 0.0) + mix_weights[d.name]

        for d in self.domains:
            if d.name not in self.effective_w:
                self.effective_w[d.name] = 1.0

                                      
        self._build_and_shuffle_index_map()

                        
        self.name2spec = {d.name: d for d in self.domains}
        self.other_domain_cache: Dict[str, List[str]] = {}
        for d in self.domains:
            self.other_domain_cache[d.name] = [x.name for x in self.domains if x.name != d.name and len(self.domain_items[x.name]) > 0]

    def _build_and_shuffle_index_map(self):
        max_len = max((len(self.domain_items[d.name]) for d in self.domains if len(self.domain_items[d.name]) > 0), default=0)
        tiles: List[Tuple[str, int]] = []
        for d in self.domains:
            items = self.domain_items[d.name]
            if not items:
                continue
            reps = max(1, int(math.ceil(self.effective_w[d.name] * max_len / max(1, len(items)))))
            pool = [(d.name, i) for i in range(len(items))]
            tiles += (pool * reps)
        self.index_map = tiles
        self.rng.shuffle(self.index_map)


    def set_epoch(self, epoch: int, mix_weights: Optional[Dict[str, float]] = None,
                  cross_domain_p: Optional[float] = None,
                  mode: Optional[Literal["inpaint","swap"]] = None,
                  self_pair_prob: Optional[float] = None):
        self.epoch = int(epoch)

        if mix_weights is not None:
            self.effective_w.clear()
            group_members: Dict[str, List[str]] = {}
            for d in self.domains:
                if d.group:
                    group_members.setdefault(d.group, []).append(d.name)
            for g, members in group_members.items():
                if g in mix_weights:
                    w_each = mix_weights[g] / max(1, len(members))
                    for m in members:
                        self.effective_w[m] = self.effective_w.get(m, 0.0) + w_each
            for d in self.domains:
                if d.name in mix_weights:
                    self.effective_w[d.name] = self.effective_w.get(d.name, 0.0) + mix_weights[d.name]
            for d in self.domains:
                if d.name not in self.effective_w:
                    self.effective_w[d.name] = 1.0
        if cross_domain_p is not None:
            self.cross_domain_p = float(max(0.0, min(1.0, cross_domain_p)))
        if mode is not None:
            self.mode = mode
        if self_pair_prob is not None:
            self.self_pair_prob = float(max(0.0, min(1.0, self_pair_prob)))

                          
        seed = (self.base_seed + 1000 * self.epoch) % (2**31 - 1)
        self.rng = random.Random(seed)
        self._build_and_shuffle_index_map()

    def __len__(self):
        return len(self.index_map)

                                
    def _clip_paths(self, dom: DomainSpec, stem: str) -> Tuple[str, str]:
        return (
            os.path.join(dom.clip_dir, f"{stem}.pt"),
            os.path.join(dom.clip_dir, f"{stem}.npy"),
        )

    def _arc_path(self, dom: DomainSpec, stem: str) -> str:
        return os.path.join(dom.arcface_dir, f"{stem}.npy")

    def _arc_full_path(self, dom: DomainSpec, stem: str) -> str:
        return os.path.join(dom.whole_arcface_dir, f"{stem}.npy")

    def _lmk_path(self, dom: DomainSpec, stem: str) -> Optional[str]:
        if not dom.lmk_dir: return None
        p = os.path.join(dom.lmk_dir, f"{stem}.txt")
        return p if os.path.exists(p) else None

    def _mask_path(self, dom: DomainSpec, stem: str) -> Optional[str]:
        if not dom.mask_dir: return None
        for ext in [".png", ".jpg", ".jpeg"]:
            p = os.path.join(dom.mask_dir, f"{stem}{ext}")
            if os.path.exists(p):
                return p
        return None

                                     
    def _sample_source(self, tgt_dom: str, tgt_idx: int, tgt_arc: Optional[np.ndarray]) -> Tuple[str, int]:
                                                  
        if self.mode == "swap" and self.self_pair_prob > 0.0:
            if self.rng.random() < self.self_pair_prob:
                return tgt_dom, tgt_idx

                  
        if self.mode == "swap" and self.rng.random() < self.cross_domain_p and len(self.other_domain_cache[tgt_dom]) > 0:
            s_dom = self.rng.choice(self.other_domain_cache[tgt_dom])
            s_size = len(self.domain_items[s_dom])
            s_idx = self.rng.randrange(0, s_size)
        else:
            s_dom = tgt_dom
            size = len(self.domain_items[s_dom])
            if size <= 1:
                return tgt_dom, tgt_idx
                                                     
            r = self.rng.randrange(0, size - 1)
            s_idx = r if r < tgt_idx else r + 1


                             
        if self.avoid_same_identity and tgt_arc is not None:
            tries = 0
            while tries < 10:
                sd = self.name2spec[s_dom]
                s_path = self.domain_items[s_dom][s_idx]
                s_stem = _stem(s_path)
                s_arc = _np_load(self._arc_path(sd, s_stem)).astype(np.float32)
                if _cosine(tgt_arc, s_arc) < self.same_id_cos_thr:
                    break
                                   
                if self.rng.random() < self.cross_domain_p and len(self.other_domain_cache[tgt_dom]) > 0:
                    s_dom = self.rng.choice(self.other_domain_cache[tgt_dom])
                    s_size = len(self.domain_items[s_dom])
                    s_idx = self.rng.randrange(0, s_size)
                else:
                    size = len(self.domain_items[s_dom])
                    r = self.rng.randrange(0, size - 1)
                    s_idx = r if r < tgt_idx else r + 1
                tries += 1

        return s_dom, s_idx

    def __getitem__(self, gidx: int) -> Dict[str, Any]:
        tgt_dom, tgt_i = self.index_map[gidx]
        td = self.name2spec[tgt_dom]
        tgt_path = self.domain_items[tgt_dom][tgt_i]
        tgt_stem = _stem(tgt_path)

                  
        tgt_img = _read_img(tgt_path, self.tf)
        pixel_values = tgt_img
        conditioning_pixel_values = tgt_img

        clip_pt, clip_npy = self._clip_paths(td, tgt_stem)
        clip_embedding = _load_tensor_pt_or_npy(clip_pt, clip_npy)

        lmk_p = self._lmk_path(td, tgt_stem)
        coords = np.loadtxt(lmk_p).reshape(-1, 2).astype(np.float32) if lmk_p else np.zeros((0, 2), dtype=np.float32)

        mask_p = self._mask_path(td, tgt_stem)
        mask = _read_mask(mask_p, self.tf) if mask_p else None

        tgt_arc_np = _np_load(self._arc_path(td, tgt_stem)).astype(np.float32) if self.avoid_same_identity else None

        if self.mode == "inpaint":
            src_dom, src_i = tgt_dom, tgt_i
        else:
            src_dom, src_i = self._sample_source(tgt_dom, tgt_i, tgt_arc_np)

        sd = self.name2spec[src_dom]
        src_path = self.domain_items[src_dom][src_i]
        src_stem = _stem(src_path)

        src_img = _read_img(src_path, self.tf)
        src_arc = torch.tensor(_np_load(self._arc_path(sd, src_stem)), dtype=torch.float32)
        src_arc_full = torch.tensor(_np_load(self._arc_full_path(sd, src_stem)), dtype=torch.float32)

        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "clip_embedding": clip_embedding,
            "coord_list": coords,
            "mask": mask,
            "source_images": src_img,
            "source_id_embeddings": src_arc,
            "source_full_embeddings": src_arc_full,
                
            "dataset_tgt": tgt_dom, "dataset_src": src_dom,
            "stem_tgt": tgt_stem, "stem_src": src_stem,
        }


                                                        
from typing import Dict, List, Optional, Tuple, Any, Literal
import os, glob, math, random
import numpy as np
import torch
from torch.utils.data import Dataset

class _SingleDomainPairDatasetBase(Dataset):

    def __init__(
        self,
        domain,                                         
        split: Literal["train","test"],
        transform,
        mode: Literal["inpaint","swap"] = "swap",
        pair_mode: Literal["rule","random"] = "random",
        self_pair_prob: float = 0.0,                                    
        specified_pair_prob: float = 1,                                     
        fallback_when_rule_fail: Literal["self","random"] = "self",
        avoid_same_identity: bool = False,
        same_id_cos_thr: float = 0.8,
        seed: int = 42,
        recursive_scan: bool = True,       
    ):
        assert split in ["train", "test"]
        assert mode in ["inpaint", "swap"]
        assert pair_mode in ["rule", "random"]
        assert fallback_when_rule_fail in ["self","random"]

        self.dom = domain
        self.split = split
        self.tf = transform

        self.mode = mode
        self.pair_mode = pair_mode
        self.self_pair_prob = float(max(0.0, min(1.0, self_pair_prob)))
        self.specified_pair_prob = float(max(0.0, min(1.0, specified_pair_prob)))
        self.fallback_when_rule_fail = fallback_when_rule_fail

        self.avoid_same_identity = avoid_same_identity
        self.same_id_cos_thr = float(same_id_cos_thr)

        self.recursive_scan = bool(recursive_scan)

              
        self.base_seed = int(seed)
        self.epoch = 0
        self.rng = random.Random(self.base_seed)

                                               
        self.items: List[str] = self._scan_images(self.dom, self.recursive_scan)
        need_train = self.dom.train_count
        need_test  = self.dom.test_count
        if need_train + need_test > 0:
            assert len(self.items) >= need_train + need_test,\
                f"[{self.dom.name}] images {len(self.items)} < train({need_train}) + test({need_test})"

        if self.split == "train":
            self.items = self.items[:need_train] if need_train > 0 else self.items
        else:
            self.items = self.items[-need_test:] if need_test > 0 else []

        assert len(self.items) > 0, f"[{self.dom.name}] no images for split={self.split}"

                    
        self._build_and_shuffle_index_map()

    def _scan_images(self, dom, recursive: bool) -> List[str]:
        paths: List[str] = []
        for ext in (dom.img_exts or IMG_EXTS):
            pattern = os.path.join(dom.image_dir, "**", f"*{ext}") if recursive else os.path.join(dom.image_dir, f"*{ext}")
            paths += glob.glob(pattern, recursive=recursive)
        def _numkey(p):
            s = _stem(p)
            try:    return int(s)
            except: return s
        return sorted(paths, key=_numkey)

    def _build_and_shuffle_index_map(self):
                                                           
        self.index_map: List[Tuple[str,int]] = [(self.dom.name, i) for i in range(len(self.items))]
        self.rng.shuffle(self.index_map)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        seed = (self.base_seed + 1000 * self.epoch) % (2**31 - 1)
        self.rng = random.Random(seed)
        self._build_and_shuffle_index_map()

    def __len__(self):
        return len(self.index_map)

                                         
    def _clip_paths(self, stem: str) -> Tuple[str, str]:
        return (
            os.path.join(self.dom.clip_dir, f"{stem}.pt"),
            os.path.join(self.dom.clip_dir, f"{stem}.npy"),
        )

    def _arc_path(self, stem: str) -> str:
        return os.path.join(self.dom.arcface_dir, f"{stem}.npy")

    def _arc_full_path(self, stem: str) -> str:
        return os.path.join(self.dom.whole_arcface_dir, f"{stem}.npy")

    def _lmk_path(self, stem: str) -> Optional[str]:
        if not self.dom.lmk_dir: return None
        p = os.path.join(self.dom.lmk_dir, f"{stem}.txt")
        return p if os.path.exists(p) else None

    def _mask_path(self, stem: str) -> Optional[str]:
        if not self.dom.mask_dir: return None
        for ext in [".png", ".jpg", ".jpeg"]:
            p = os.path.join(self.dom.mask_dir, f"{stem}{ext}")
            if os.path.exists(p):
                return p
        return None

                                         
    def _rule_pair_index(self, n: int, idx: int) -> int:
        half = n // 2
                   
                                        
                                             
                                             
        return (idx + half) if (idx < half) else (idx - half)

                                     
    def _sample_source_index(self, n: int, tgt_idx: int, tgt_arc: Optional[np.ndarray]) -> int:
                     
        if self.mode == "inpaint":
            return tgt_idx

                             
        if self.rng.random() < self.self_pair_prob:
            s_idx = tgt_idx
        else:
                                 
            if self.pair_mode == "rule":
                                          
                use_rule = (self.rng.random() < self.specified_pair_prob)
                if use_rule and n >= 2:
                    s_idx = self._rule_pair_index(n, tgt_idx)
                else:
                                               
                    s_idx = self._uniform_diff_index(n, tgt_idx)
            else:
                                                    
                s_idx = self._uniform_diff_index(n, tgt_idx)

                            
        if self.avoid_same_identity and tgt_arc is not None:
            tries = 0
            while tries < 10:
                s_path = self.items[s_idx]
                s_stem = _stem(s_path)
                s_arc = _np_load(self._arc_path(s_stem)).astype(np.float32)
                if _cosine(tgt_arc, s_arc) < self.same_id_cos_thr:
                    break
                s_idx = self._uniform_diff_index(n, tgt_idx)              
                tries += 1

        return s_idx

    def _uniform_diff_index(self, n: int, exclude_idx: int) -> int:
        if n <= 1: return exclude_idx
        r = self.rng.randrange(0, n - 1)
        return r if r < exclude_idx else r + 1

                              
    def __getitem__(self, gidx: int) -> Dict[str, Any]:
        _, tgt_i = self.index_map[gidx]
        tgt_path = self.items[tgt_i]
        tgt_stem = _stem(tgt_path)

                  
        tgt_img = _read_img(tgt_path, self.tf)
        pixel_values = tgt_img
        conditioning_pixel_values = tgt_img

        clip_pt, clip_npy = self._clip_paths(tgt_stem)
        clip_embedding = _load_tensor_pt_or_npy(clip_pt, clip_npy)

        lmk_p = self._lmk_path(tgt_stem)
        coords = np.loadtxt(lmk_p).reshape(-1, 2).astype(np.float32) if lmk_p else np.zeros((0, 2), dtype=np.float32)

        mask_p = self._mask_path(tgt_stem)
        mask = _read_mask(mask_p, self.tf) if mask_p else None

        tgt_arc_np = _np_load(self._arc_path(tgt_stem)).astype(np.float32) if self.avoid_same_identity else None

                
        s_i = self._sample_source_index(len(self.items), tgt_i, tgt_arc_np)
        src_path = self.items[s_i]
        src_stem = _stem(src_path)

        src_img = _read_img(src_path, self.tf)
        src_arc = torch.tensor(_np_load(self._arc_path(src_stem)), dtype=torch.float32)
        src_arc_full = torch.tensor(_np_load(self._arc_full_path(src_stem)), dtype=torch.float32)

        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "clip_embedding": clip_embedding,
            "coord_list": coords,
            "mask": mask,
            "source_images": src_img,
            "source_id_embeddings": src_arc,
            "source_full_embeddings": src_arc_full,
                
            "dataset_tgt": self.dom.name, "dataset_src": self.dom.name,
            "stem_tgt": tgt_stem, "stem_src": src_stem,
        }

                                                       
class SingleFFHQPairDataset(_SingleDomainPairDatasetBase):

    def __init__(self, domain, *args, **kwargs):
        kwargs.setdefault("recursive_scan", True)             
        super().__init__(domain, *args, **kwargs)

class SingleCelebAPairDataset(_SingleDomainPairDatasetBase):

    def __init__(self, domain, *args, **kwargs):
        kwargs.setdefault("recursive_scan", True)               
        super().__init__(domain, *args, **kwargs)


                                                                              

class Stage1CelebAAugDataset(Dataset):

    def __init__(
        self,
        *,
        image_dir: str,                            
        aug_image_dir: str,                            
        clip_dir: str,                                                    
        arcface_dir: str,                                    
        whole_arcface_dir: str,                                    
        lmk_dir: Optional[str] = None,                            
        mask_dir: Optional[str] = None,                  
        transform=None,
        train_count: int = 28000,                 
        id_offset: int = 30000,                             
        img_exts: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.image_dir = image_dir
        self.aug_image_dir = aug_image_dir
        self.clip_dir = clip_dir
        self.arcface_dir = arcface_dir
        self.whole_arcface_dir = whole_arcface_dir
        self.lmk_dir = lmk_dir
        self.mask_dir = mask_dir
        self.tf = transform
        self.id_offset = int(id_offset)
        self.rng = random.Random(int(seed))
        self.img_exts = img_exts or IMG_EXTS

                                                    
        tgt_paths = []
        for ext in self.img_exts:
            tgt_paths += glob.glob(os.path.join(self.image_dir, "**", f"*{ext}"), recursive=True)

        def _numkey(p):
            s = _stem(p)
            try:
                return int(s)
            except:
                return s
        tgt_paths = sorted(tgt_paths, key=_numkey)
        if train_count > 0:
            tgt_paths = tgt_paths[:train_count]
        self.tgt_paths = tgt_paths

                                                  
        self.tgt_stem2path = { _stem(p): p for p in self.tgt_paths }

        aug_paths = []
        for ext in self.img_exts:
            aug_paths += glob.glob(os.path.join(self.aug_image_dir, "**", f"*{ext}"), recursive=True)
        self.aug_stem2path = { _stem(p): p for p in aug_paths }

                              
        self._warned_missing_src = False

                                                                   
        self.base_seed = int(seed)
        self.epoch = 0
        self.rng = random.Random(self.base_seed)
        self.index_map = list(range(len(self.tgt_paths)))                      


    def __len__(self):
        return len(self.index_map)

    def set_epoch(
        self,
        epoch: int,
        mix_weights: Optional[Dict[str, float]] = None,
        cross_domain_p: Optional[float] = None,
        mode: Optional[Literal["inpaint", "swap"]] = None,
        self_pair_prob: Optional[float] = None,
    ):
        """与 MultiDomainPairDataset 接口保持一致；
        这里只用来按 epoch 重建并洗牌 index_map，确保 DDP 下各 rank 顺序一致。"""
        self.epoch = int(epoch)
        seed = (self.base_seed + 1000 * self.epoch) % (2**31 - 1)
        self.rng = random.Random(seed)
        self.index_map = list(range(len(self.tgt_paths)))
        self.rng.shuffle(self.index_map)

              
    def _clip_paths(self, stem: str) -> Tuple[str, str]:
        return (
            os.path.join(self.clip_dir, f"{stem}.pt"),
            os.path.join(self.clip_dir, f"{stem}.npy"),
        )

    def _arc_path(self, stem: str) -> str:
        return os.path.join(self.arcface_dir, f"{stem}.npy")

    def _arc_full_path(self, stem: str) -> str:
        return os.path.join(self.whole_arcface_dir, f"{stem}.npy")

    def _lmk_path(self, stem: str) -> Optional[str]:
        if not self.lmk_dir:
            return None
        p = os.path.join(self.lmk_dir, f"{stem}.txt")
        return p if os.path.exists(p) else None

    def _mask_path(self, stem: str) -> Optional[str]:
        if not self.mask_dir:
            return None
        for ext in [".png", ".jpg", ".jpeg"]:
            p = os.path.join(self.mask_dir, f"{stem}{ext}")
            if os.path.exists(p):
                return p
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = self.index_map[idx]
        tgt_path = self.tgt_paths[idx]
        tgt_stem = _stem(tgt_path)

                          
        tgt_img = _read_img(tgt_path, self.tf)
        pixel_values = tgt_img
        conditioning_pixel_values = tgt_img

        clip_pt, clip_npy = self._clip_paths(tgt_stem)
        clip_embedding = _load_tensor_pt_or_npy(clip_pt, clip_npy)

        lmk_p = self._lmk_path(tgt_stem)
        coords = np.loadtxt(lmk_p).reshape(-1, 2).astype(np.float32) if lmk_p else np.zeros((0, 2), dtype=np.float32)

        mask_p = self._mask_path(tgt_stem)
        mask = _read_mask(mask_p, self.tf) if mask_p else None

                                                                            
        src_stem = tgt_stem
        try:
                                  
            width = len(tgt_stem) if tgt_stem.isdigit() and tgt_stem.startswith("0") else 0
            num = int(tgt_stem)
            src_num = num + self.id_offset
            src_stem_try = str(src_num).zfill(width) if width > 0 else str(src_num)
            if src_stem_try in self.aug_stem2path:
                src_stem = src_stem_try
                src_path = self.aug_stem2path[src_stem]
            else:
                                         
                src_path = tgt_path
                if not self._warned_missing_src:
                    print(f"[Stage1CelebAAugDataset] WARN: mapped src not found for '{tgt_stem}' "
                          f"-> '{src_stem_try}', fallback to self-pair once.")
                    self._warned_missing_src = True
        except ValueError:
                                 
            src_path = tgt_path
            if not self._warned_missing_src:
                print(f"[Stage1CelebAAugDataset] WARN: non-numeric stem '{tgt_stem}', fallback to self-pair once.")
                self._warned_missing_src = True

        src_img = _read_img(src_path, self.tf)

                     
        src_arc = torch.tensor(_np_load(self._arc_path(src_stem)), dtype=torch.float32)
        src_arc_full = torch.tensor(_np_load(self._arc_full_path(src_stem)), dtype=torch.float32)

        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "clip_embedding": clip_embedding,
            "coord_list": coords,
            "mask": mask,
            "source_images": src_img,
            "source_id_embeddings": src_arc,
            "source_full_embeddings": src_arc_full,
                  
            "dataset_tgt": "celeba",
            "dataset_src": "celeba_aug",
            "stem_tgt": tgt_stem,
            "stem_src": src_stem,
        }


def _stack_maybe_squeeze_first_dim(t_list):

    fixed = []
    for t in t_list:
        if t is None:
            fixed.append(None)
            continue
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        if t.ndim >= 1 and t.shape[0] == 1:
            t = t.squeeze(0)
        fixed.append(t)

                      
    shape = None
    for t in fixed:
        if t is not None:
            shape = t.shape
            break
    fixed = [torch.zeros(shape, dtype=torch.float32) if t is None else t.float() for t in fixed]
    return torch.stack(fixed, dim=0)


def collate_fn(examples):

             
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples], dim=0).float()
    conditioning_pixel_values = torch.stack([ex["conditioning_pixel_values"] for ex in examples], dim=0).float()
    source_images = torch.stack([ex["source_images"] for ex in examples], dim=0).float()
                                                    
    mask_list = [ex["mask"] for ex in examples]
    if all(m is None for m in mask_list):
        B, _, H, W = pixel_values.shape                              
        masks = torch.zeros((B, 1, H, W), dtype=torch.float32)
    else:
        masks = _stack_maybe_squeeze_first_dim(mask_list)


                       
    clip_embeddings = _stack_maybe_squeeze_first_dim([ex["clip_embedding"] for ex in examples])
    source_id_embeddings = _stack_maybe_squeeze_first_dim([ex["source_id_embeddings"] for ex in examples])
    source_full_embeddings = _stack_maybe_squeeze_first_dim([ex["source_full_embeddings"] for ex in examples])

                           
    coord_list = torch.stack([torch.tensor(ex["coord_list"], dtype=torch.float32) for ex in examples], dim=0)

    batch = {
        "pixel_values": pixel_values.to(memory_format=torch.contiguous_format),
        "conditioning_pixel_values": conditioning_pixel_values.to(memory_format=torch.contiguous_format),
        "clip_embedding": clip_embeddings,
        "source_id_embeddings": source_id_embeddings,
        "source_full_embeddings": source_full_embeddings,
        "source_images": source_images.to(memory_format=torch.contiguous_format),
        "coord_list": coord_list,
        "mask": masks.to(memory_format=torch.contiguous_format),
    }

                                          
    metas = []
    for ex in examples:
        if ("dataset_tgt" in ex) and ("dataset_src" in ex) and ("stem_tgt" in ex) and ("stem_src" in ex):
            metas.append({
                "tgt_dom": ex["dataset_tgt"],
                "src_dom": ex["dataset_src"],
                "tgt_stem": ex["stem_tgt"],
                "src_stem": ex["stem_src"],
            })
    if metas:
        batch["__meta__"] = metas

    return batch