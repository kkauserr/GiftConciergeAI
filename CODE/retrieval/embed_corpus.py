"""Encode `embed_text` for every product into sharded, normalized embeddings.

Streams the corpus line by line (never loads the 27 GB file into memory) and
writes one shard every `--shard-size` products:

    <out-dir>/emb_00000.npy     float32 [n, 384], L2-normalized
    <out-dir>/asins_00000.txt   parent_asin per row, same order
    <out-dir>/progress.json     resume checkpoint (byte offset + shard count)

Resumable: rerunning the same command continues from the last completed
shard, so a Sol job that hits its time limit can simply be resubmitted.
After all shards are written, run build_faiss_index.py.

Local (sample):
    python CODE/retrieval/embed_corpus.py

Sol (full corpus):
    python CODE/retrieval/embed_corpus.py \
        --input /scratch/jkauser/gift-concierge/data/final_product_corpus_v2.jsonl \
        --out-dir /scratch/$USER/gift-concierge/embeddings/full_v2 \
        --batch-size 512 --device cuda --fp16
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.config import (  # noqa: E402
    EMBED_DIM,
    MODEL_NAME,
    PROGRESS_FILE,
    SAMPLE_CORPUS,
    SAMPLE_INDEX_DIR,
    SHARD_ASINS,
    SHARD_EMB,
    TEXT_FIELD,
)


def product_text(record):
    """Text to embed; falls back to the title so every product gets a row."""
    text = (record.get(TEXT_FIELD) or "").strip()
    return text or (record.get("title") or "").strip()


def load_encoder(model_name=MODEL_NAME, device=None, fp16=False):
    """Return encode(texts, batch_size) -> float32 [n, dim] normalized embeddings."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    if fp16 and str(model.device).startswith("cuda"):
        model.half()

    def encode(texts, batch_size=64):
        vectors = model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    return encode


def _write_json_atomic(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _save_shard(out_dir, shard_id, vectors, asins):
    """Write shard files atomically (tmp file + rename) before progress moves on."""
    emb_path = out_dir / SHARD_EMB.format(shard_id)
    asin_path = out_dir / SHARD_ASINS.format(shard_id)
    tmp_emb = out_dir / f"tmp_{SHARD_EMB.format(shard_id)}"
    np.save(tmp_emb, vectors)
    os.replace(tmp_emb, emb_path)
    tmp_asin = Path(str(asin_path) + ".tmp")
    tmp_asin.write_text("\n".join(asins) + "\n", encoding="utf-8")
    os.replace(tmp_asin, asin_path)


def _load_progress(out_dir, config):
    path = out_dir / PROGRESS_FILE
    if not path.exists():
        return {
            **config,
            "byte_offset": 0,
            "completed_shards": 0,
            "rows_done": 0,
            "skipped_lines": 0,
            "finished": False,
        }
    progress = json.loads(path.read_text(encoding="utf-8"))
    for key, value in config.items():
        if progress.get(key) != value:
            raise ValueError(
                f"{path} was created with {key}={progress.get(key)!r}, "
                f"but this run uses {key}={value!r}. Use a new --out-dir."
            )
    return progress


def embed_corpus(
    input_path,
    out_dir,
    model_name=MODEL_NAME,
    batch_size=256,
    shard_size=100_000,
    device=None,
    fp16=False,
    max_shards=None,
    encoder=None,
    log=print,
):
    """Embed the corpus into shards, resuming from progress.json if present.

    `encoder` can be injected (tests); otherwise the sentence-transformers
    model is loaded. `max_shards` stops after that many new shards, which is
    handy for time-boxed jobs. Returns the progress dict.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "input": str(input_path),
        "model": model_name,
        "text_field": TEXT_FIELD,
        "shard_size": shard_size,
        "normalized": True,
    }
    progress = _load_progress(out_dir, config)
    if progress["finished"]:
        log(f"Already finished: {progress['rows_done']} rows in "
            f"{progress['completed_shards']} shards.")
        return progress

    if progress["rows_done"]:
        log(f"Resuming at shard {progress['completed_shards']} "
            f"(row {progress['rows_done']}, byte {progress['byte_offset']}).")

    if encoder is None:
        encoder = load_encoder(model_name, device=device, fp16=fp16)

    start_time = time.time()
    new_shards = 0
    texts, asins = [], []
    skipped = 0

    def flush(end_offset):
        nonlocal texts, asins, skipped
        vectors = encoder(texts, batch_size)
        if vectors.shape != (len(texts), EMBED_DIM):
            raise ValueError(f"Unexpected embedding shape {vectors.shape}")
        shard_id = progress["completed_shards"]
        _save_shard(out_dir, shard_id, vectors, asins)
        progress["completed_shards"] += 1
        progress["rows_done"] += len(asins)
        progress["skipped_lines"] += skipped
        progress["byte_offset"] = end_offset
        _write_json_atomic(out_dir / PROGRESS_FILE, progress)
        log(f"shard {shard_id:05d}: {len(asins)} rows "
            f"(total {progress['rows_done']}, {time.time() - start_time:.0f}s)")
        texts, asins, skipped = [], [], 0

    with open(input_path, "rb") as infile:
        infile.seek(progress["byte_offset"])
        offset = progress["byte_offset"]

        for raw in infile:
            offset += len(raw)
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
                asin = record["parent_asin"]
            except (json.JSONDecodeError, KeyError, TypeError):
                skipped += 1
                continue
            texts.append(product_text(record))
            asins.append(asin)

            if len(asins) >= shard_size:
                flush(offset)
                new_shards += 1
                if max_shards is not None and new_shards >= max_shards:
                    log(f"Stopped after {new_shards} new shard(s); rerun to resume.")
                    return progress

    if asins:
        flush(offset)
    else:
        progress["skipped_lines"] += skipped
        progress["byte_offset"] = offset

    progress["finished"] = True
    _write_json_atomic(out_dir / PROGRESS_FILE, progress)
    log(f"Finished: {progress['rows_done']} rows, {progress['completed_shards']} "
        f"shards, {progress['skipped_lines']} unparseable lines skipped.")
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", default=str(SAMPLE_CORPUS))
    parser.add_argument("--out-dir", default=str(SAMPLE_INDEX_DIR))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--fp16", action="store_true", help="half precision on GPU")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="stop after N new shards (rerun to resume)")
    args = parser.parse_args()

    embed_corpus(
        input_path=args.input,
        out_dir=args.out_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        device=args.device,
        fp16=args.fp16,
        max_shards=args.max_shards,
    )


if __name__ == "__main__":
    main()
