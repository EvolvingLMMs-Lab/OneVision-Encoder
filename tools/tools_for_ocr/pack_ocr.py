import os
import json
import pathlib
import numpy as np
np.bool = np.bool_  # 解决 mxnet 与 numpy 冲突

import mxnet as mx
import mxnet.recordio as recordio

ROOTS = [pathlib.Path("/data_2"), pathlib.Path("/data_3")]
JSONL_SUFFIX = ".jsonl"
LANG_PREFIXES = ("en_", "zh_")
TARGET_LEN = 100

def get_indexed_recordio(idx_path, rec_path, flag):
    if hasattr(recordio, "IndexedRecordIO"):
        cls = recordio.IndexedRecordIO
    else:
        cls = recordio.MXIndexedRecordIO
    return cls(str(idx_path), str(rec_path), flag)

def iter_jsonl_files():
    for root in ROOTS:
        if not root.exists():
            continue
        for p in root.rglob(f"*{JSONL_SUFFIX}"):
            yield p

def strip_lang_prefix(name: str) -> str:
    for pref in LANG_PREFIXES:
        if name.startswith(pref):
            return name[len(pref):]
    return name

def pad_or_trunc_label(label):
    if not label:
        return [0] * TARGET_LEN
    if len(label) >= TARGET_LEN:
        return label[:TARGET_LEN]
    out = []
    i = 0
    while len(out) < TARGET_LEN:
        out.append(label[i % len(label)])
        i += 1
    return out

def process_one_jsonl(jsonl_path: pathlib.Path):
    print(f"\n[INFO] processing jsonl: {jsonl_path}")
    lang_dir = jsonl_path.parent
    base_dir = lang_dir.parent
    chunk_with_lang = lang_dir.name
    chunk = strip_lang_prefix(chunk_with_lang)

    rec_dir = base_dir / chunk
    rec_path = rec_dir / f"{chunk}.rec"
    idx_path = rec_dir / f"{chunk}.idx"

    print(f"[INFO] expect rec/idx in: {rec_dir}")
    if not rec_path.exists() or not idx_path.exists():
        print(f"[WARN] rec/idx missing for {jsonl_path}")
        return

    out_dir = base_dir / f"{chunk}_ocr_labeled"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rec = out_dir / f"{chunk}_ocr_labeled.rec"
    out_idx = out_dir / f"{chunk}_ocr_labeled.idx"
    print(f"[INFO] writing to: {out_rec}")

    rec_in = get_indexed_recordio(idx_path, rec_path, "r")
    rec_out = get_indexed_recordio(out_idx, out_rec, "w")

    total = 0
    written = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rec_id = int(obj["record_id"])
                label = pad_or_trunc_label(obj.get("gt", []))
            except Exception as e:
                print(f"[WARN] parse jsonl failed {jsonl_path}:{line_no} -> {e}")
                continue

            raw = rec_in.read_idx(rec_id)
            if raw is None:
                print(f"[WARN] record_id {rec_id} not found in {rec_path}")
                continue

            header, img = recordio.unpack(raw)
            new_header = recordio.IRHeader(
                flag=header.flag if hasattr(header, "flag") else 0,
                label=label,
                id=rec_id,
                id2=header.id2 if hasattr(header, "id2") else 0,
            )
            packed = recordio.pack(new_header, img)
            rec_out.write_idx(rec_id, packed)
            written += 1

            if written % 1000 == 0:
                print(f"[INFO] {jsonl_path.name}: written {written} / {total} processed lines")

    rec_in.close()
    rec_out.close()
    print(f"[DONE] {jsonl_path} -> {out_rec}, written {written} / {total} lines")

def main():
    jsonls = list(iter_jsonl_files())
    print(f"[INFO] found {len(jsonls)} jsonl files")
    for jsonl in jsonls:
        process_one_jsonl(jsonl)

if __name__ == "__main__":
    main()
