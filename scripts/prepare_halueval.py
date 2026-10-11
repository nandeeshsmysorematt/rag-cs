#!/usr/bin/env python
"""Convert HaluEval-QA (qa_data.json, JSON lines with knowledge / question /
right_answer / hallucinated_answer) into the unified RAG-CS JSONL format:
two examples per row (label 0 = right answer, label 1 = hallucinated answer)."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--keep-knowledge", action="store_true",
                    help="attach HaluEval's 'knowledge' field (for --use-example-knowledge)")
    args = ap.parse_args()

    n = 0
    with open(args.input, encoding="utf-8") as f, open(args.output, "w", encoding="utf-8") as out:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            for suffix, key, label in (("right", "right_answer", 0), ("hall", "hallucinated_answer", 1)):
                ex = {"id": f"halueval-qa-{i}-{suffix}", "query": row["question"],
                      "response": row[key], "label": label}
                if args.keep_knowledge and row.get("knowledge"):
                    ex["knowledge"] = [row["knowledge"]]
                out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} examples to {args.output}")


if __name__ == "__main__":
    main()
