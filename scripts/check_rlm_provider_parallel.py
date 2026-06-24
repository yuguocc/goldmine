from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBS_ROOT = PROJECT_ROOT / "libs"
if str(LIBS_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBS_ROOT))

from rlm.utils.llm import OpenAIClient


def _call_provider(index: int, *, model: str, prompt: str) -> dict[str, object]:
    client = OpenAIClient(model=model)
    started = time.perf_counter()
    error = ""
    response = ""
    try:
        response = client.completion(
            [
                {
                    "role": "user",
                    "content": f"{prompt}\nRequest index: {index}. Reply with one word: ok.",
                }
            ],
            max_tokens=16,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    return {
        "index": index,
        "started": started,
        "ended": ended,
        "elapsed": ended - started,
        "ok": not error,
        "error": error,
        "response_preview": str(response or "")[:120],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether the configured RLM provider handles concurrent calls."
    )
    parser.add_argument("--model", default="z-ai/glm-5")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument(
        "--prompt",
        default="This is a provider concurrency probe.",
    )
    args = parser.parse_args()

    worker_count = max(1, min(args.workers, args.requests))
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_call_provider, index, model=args.model, prompt=args.prompt)
            for index in range(args.requests)
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    ended = time.perf_counter()

    ordered = sorted(results, key=lambda item: int(item["index"]))
    max_single = max((float(item["elapsed"]) for item in ordered), default=0.0)
    total_child = sum(float(item["elapsed"]) for item in ordered)
    wall = ended - started
    parallel_efficiency = (total_child / wall) if wall > 0 else 0.0
    likely_parallel = wall <= max_single * 1.6 if max_single else False

    print(
        json.dumps(
            {
                "model": args.model,
                "workers": worker_count,
                "requests": args.requests,
                "wall_elapsed": wall,
                "max_single_elapsed": max_single,
                "sum_child_elapsed": total_child,
                "parallel_efficiency": parallel_efficiency,
                "likely_parallel": likely_parallel,
                "results": ordered,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
