#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


COMPONENT_FILES = {
    "target": "llm.mnn",
    "eagle": "eagle.mnn",
    "eagle_fc": "eagle_fc.mnn",
}


def run(command, cwd=None):
    print("+", " ".join(str(arg) for arg in command))
    subprocess.run([str(arg) for arg in command], cwd=cwd, check=True)


def convert_component(args, component, cache_root, output_root):
    model_file = COMPONENT_FILES[component]
    component_cache = cache_root / component
    test_dir = component_cache / "testdir"
    qnn_relative = Path("qnn") / component
    qnn_dir = component_cache / qnn_relative
    component_cache.mkdir(parents=True, exist_ok=True)
    qnn_dir.mkdir(parents=True, exist_ok=True)

    run([
        args.mnn_path / "generateLlmIO",
        args.model,
        test_dir,
        component,
        *args.buckets,
    ])

    compile_config = {
        "type": "QNN",
        "skips": [],
        "testdir": [str(Path("testdir") / str(bucket)) for bucket in args.buckets],
        "cache": str(qnn_relative),
    }
    config_path = component_cache / "qnn.json"
    config_path.write_text(json.dumps(compile_config, indent=4) + "\n")

    run([
        args.mnn_path / "compilefornpu",
        args.model / model_file,
        qnn_relative / model_file,
        config_path.name,
    ], cwd=component_cache)

    converter = Path(__file__).resolve().parents[4] / "source" / "backend" / "qnn" / "npu_convert.py"
    run([
        sys.executable,
        converter,
        "npu_postreat.json",
        args.soc_id,
        args.dsp_arch,
    ], cwd=component_cache)

    component_output = output_root / component
    if component_output.exists():
        shutil.rmtree(component_output)
    shutil.copytree(qnn_dir, component_output)


def write_runtime_config(args):
    base_config_path = args.base_config or args.model / "config.json"
    config = {}
    if base_config_path.exists():
        config = json.loads(base_config_path.read_text())
    config.update({
        "llm_model": "qnn/target/llm.mnn",
        "eagle_model": "qnn/eagle/eagle.mnn",
        "eagle_fc": "qnn/eagle_fc/eagle_fc.mnn",
        "eagle_d2t": "eagle_d2t.mnn",
        "backend_type": args.backend_type,
        "thread_num": args.thread_num,
        "packed_attention_mode": True,
        "dual_pipeline_max_resident_graphs": args.dual_pipeline_max_resident_graphs,
        "dual_pipeline_prefetch_window": args.dual_pipeline_prefetch_window,
        "chunk_limits": sorted(args.buckets, reverse=True),
        "precision": args.precision,
        "memory": args.memory,
        "speculative_type": "eagle",
        "hidden_states": True,
    })
    (args.model / "config_qnn.json").write_text(json.dumps(config, indent=4) + "\n")


def convert(args):
    cache_root = args.cache_path
    staging_output = cache_root / "output_qnn"
    if cache_root.exists():
        shutil.rmtree(cache_root)
    staging_output.mkdir(parents=True)

    for component in args.components:
        convert_component(args, component, cache_root, staging_output)

    final_output = args.model / "qnn"
    if final_output.exists():
        shutil.rmtree(final_output)
    shutil.move(str(staging_output), final_output)
    write_runtime_config(args)
    shutil.rmtree(cache_root)
    print("Generated QNN components in", final_output)


def main():
    parser = argparse.ArgumentParser(description="Generate target and Eagle3 QNN offline caches")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--soc_id", type=int, required=True)
    parser.add_argument("--dsp_arch", required=True)
    parser.add_argument("--mnn_path", type=Path, default=Path("../../../build"))
    parser.add_argument("--cache_path", type=Path, default=Path("tmp_qnn_eagle3"))
    parser.add_argument("--components", nargs="+", choices=COMPONENT_FILES, default=list(COMPONENT_FILES))
    parser.add_argument("--buckets", nargs="+", type=int, default=[1, 32, 256, 512])
    parser.add_argument("--base_config", type=Path)
    parser.add_argument("--backend_type", choices=["cpu", "opencl"], default="opencl")
    parser.add_argument("--thread_num", type=int, default=1)
    parser.add_argument("--precision", choices=["low", "high"], default="low")
    parser.add_argument("--memory", choices=["low", "high"], default="low")
    parser.add_argument("--dual_pipeline_max_resident_graphs", type=int, default=2)
    parser.add_argument("--dual_pipeline_prefetch_window", type=int, default=0)
    args = parser.parse_args()

    args.model = args.model.resolve()
    args.mnn_path = args.mnn_path.resolve()
    args.cache_path = args.cache_path.resolve()
    args.buckets = sorted(set(args.buckets))
    if args.buckets != [1, 32, 256, 512]:
        parser.error("QNN buckets are fixed to 1 32 256 512")
    if args.dual_pipeline_max_resident_graphs <= 0:
        parser.error("dual pipeline resident graph limit must be positive")
    if args.dual_pipeline_prefetch_window < 0:
        parser.error("dual pipeline prefetch window must be non-negative")
    for component in args.components:
        model_path = args.model / COMPONENT_FILES[component]
        if not model_path.is_file():
            parser.error(f"missing {component} model: {model_path}")
    convert(args)


if __name__ == "__main__":
    main()
