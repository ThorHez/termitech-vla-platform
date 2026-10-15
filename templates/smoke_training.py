"""Explicit CPU-only platform smoke workload. This does not train a VLA model."""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.00005)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print("[平台链路测试] CPU 合成数据；此任务不使用 GPU，不训练 VLA 模型。", flush=True)
    weight = 0.0
    for step in range(1, args.steps + 1):
        # A tiny actual gradient update keeps the process and metric path testable.
        xs = [(i + 1) / args.batch_size for i in range(args.batch_size)]
        loss = sum((weight * x - 2 * x) ** 2 for x in xs) / len(xs)
        grad = sum(2 * (weight * x - 2 * x) * x for x in xs) / len(xs)
        weight -= args.learning_rate * grad
        print("VLA_EVENT:" + json.dumps({"type": "METRIC", "data": {
            "step": step, "total_steps": args.steps, "loss": loss,
            "learning_rate": args.learning_rate, "simulation": True}}), flush=True)
        time.sleep(0.1)
    checkpoint = output / "smoke-result"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "result.json").write_text(json.dumps({"weight": weight, "simulation": True}))
    (output / "artifacts.json").write_text(json.dumps([{
        "kind": "MODEL", "name": "链路测试产物（非 VLA 模型）", "path": str(checkpoint),
        "version": "smoke", "metadata": {"simulation": True, "format": "smoke-json", "evaluated": False}
    }], ensure_ascii=False))
    print("链路测试结束，已写入测试产物。", flush=True)


if __name__ == "__main__":
    main()
