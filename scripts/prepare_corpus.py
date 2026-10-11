#!/usr/bin/env python
"""Build the passage corpus K as JSONL ({"id","title","text"} per line).

--format articles : input JSONL of {"title","text"}; split into 100-word passages
                    with a 20-word overlap (stride 80), title kept per passage.
--format dpr_tsv  : input is DPR's psgs_w100.tsv (header: id<TAB>text<TAB>title).
"""
import argparse
import csv
import json
import sys

from tqdm import tqdm


def chunk(words, size=100, overlap=20):
    stride = size - overlap
    for start in range(0, max(len(words), 1), stride):
        piece = words[start:start + size]
        if piece:
            yield " ".join(piece)
        if start + size >= len(words):
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--format", choices=["articles", "dpr_tsv"], default="articles")
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--overlap", type=int, default=20)
    args = ap.parse_args()

    n = 0
    with open(args.output, "w", encoding="utf-8") as out:
        if args.format == "articles":
            with open(args.input, encoding="utf-8") as f:
                for line in tqdm(f, desc="articles"):
                    art = json.loads(line)
                    for text in chunk(art["text"].split(), args.size, args.overlap):
                        out.write(json.dumps({"id": n, "title": art.get("title", ""), "text": text},
                                             ensure_ascii=False) + "\n")
                        n += 1
        else:
            csv.field_size_limit(sys.maxsize)
            with open(args.input, encoding="utf-8", newline="") as f:
                rd = csv.reader(f, delimiter="\t")
                next(rd)  # header
                for row in tqdm(rd, desc="dpr_tsv"):
                    out.write(json.dumps({"id": n, "title": row[2], "text": row[1]},
                                         ensure_ascii=False) + "\n")
                    n += 1
    print(f"wrote {n} passages to {args.output}")


if __name__ == "__main__":
    main()
