from __future__ import annotations

import argparse
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import torch
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holitok.config import to_hparams
from holitok.presets import checkpoint_filename
from holitok.presets import get_preset_config
from holitok.runtime import BigVGANVAEVQDS


MODEL_CARD = """---
license: apache-2.0
library_name: pytorch
pipeline_tag: audio-to-audio
---

# HoliTok

This repository hosts runtime-only HoliTok checkpoints for inference. The
checkpoints are loaded by the HoliTok inference package through the public
aliases `HoliTok-Base` and `HoliTok-Unite`.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repo id")
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--unite-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to keep the exported runtime-only checkpoints",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=None, help="Optional HF token; otherwise uses logged-in token")
    return parser.parse_args()


def strip_module_prefix(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def export_runtime_checkpoint(
    *,
    source_path: Path,
    output_path: Path,
    model_name: str,
) -> tuple[int, int, int]:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    runtime_model = BigVGANVAEVQDS(to_hparams(get_preset_config(model_name)))
    runtime_state = runtime_model.state_dict()
    source_state = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(source_state, dict) and "model" in source_state:
        source_state = source_state["model"]
    if not isinstance(source_state, dict):
        raise TypeError(f"Expected a state dict in {source_path}, got {type(source_state)!r}")

    exported = OrderedDict()
    skipped = 0
    shape_mismatch = []
    for key, value in source_state.items():
        clean_key = strip_module_prefix(key)
        if clean_key not in runtime_state:
            skipped += 1
            continue
        if tuple(runtime_state[clean_key].shape) != tuple(value.shape):
            shape_mismatch.append((clean_key, tuple(value.shape), tuple(runtime_state[clean_key].shape)))
            continue
        exported[clean_key] = value.detach().cpu()

    missing = sorted(set(runtime_state) - set(exported))
    if missing or shape_mismatch:
        raise RuntimeError(
            f"Cannot export {model_name}: missing={len(missing)} shape_mismatch={len(shape_mismatch)} "
            f"missing_sample={missing[:8]} shape_mismatch_sample={shape_mismatch[:4]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "holitok-runtime-state-dict-v1",
            "model_name": model_name,
            "model": exported,
            "metadata": {
                "num_tensors": len(exported),
                "num_source_tensors": len(source_state),
                "num_skipped_tensors": skipped,
            },
        },
        output_path,
    )
    return len(exported), len(source_state), skipped


def upload_file(api: HfApi, *, repo_id: str, local_path: Path, path_in_repo: str, token, revision: str) -> None:
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        revision=revision,
        token=token,
    )


def main() -> None:
    args = parse_args()
    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
        token=args.token,
    )

    with tempfile.TemporaryDirectory(prefix="holitok_hf_card_") as tmp:
        card_path = Path(tmp) / "README.md"
        card_path.write_text(MODEL_CARD, encoding="utf-8")
        upload_file(
            api,
            repo_id=args.repo_id,
            local_path=card_path,
            path_in_repo="README.md",
            revision=args.revision,
            token=args.token,
        )

    if args.output_dir is None:
        export_context = tempfile.TemporaryDirectory(prefix="holitok_runtime_ckpts_")
        output_root = Path(export_context.name)
    else:
        export_context = None
        output_root = args.output_dir

    try:
        exported_paths = {}
        for model_name, source_path in (
            ("HoliTok-Base", args.base_checkpoint),
            ("HoliTok-Unite", args.unite_checkpoint),
        ):
            output_path = output_root / checkpoint_filename(model_name)
            kept, total, skipped = export_runtime_checkpoint(
                source_path=source_path,
                output_path=output_path,
                model_name=model_name,
            )
            exported_paths[model_name] = output_path
            size_gib = output_path.stat().st_size / 1024**3
            print(
                f"Exported {model_name}: kept {kept}/{total} tensors, "
                f"skipped {skipped}, size {size_gib:.3f} GiB -> {output_path}"
            )

        for model_name, local_path in exported_paths.items():
            upload_file(
                api,
                repo_id=args.repo_id,
                local_path=local_path,
                path_in_repo=checkpoint_filename(model_name),
                revision=args.revision,
                token=args.token,
            )
    finally:
        if export_context is not None:
            export_context.cleanup()

    print(f"Uploaded HoliTok checkpoints to {args.repo_id}")


if __name__ == "__main__":
    main()
