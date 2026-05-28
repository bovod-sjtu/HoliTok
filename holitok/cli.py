from __future__ import annotations

import argparse

import torch

from .io import load_audio, save_audio
from .model import HoliTok
from .presets import preset_names
from .semantic import SemanticModule


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default="HoliTok-Unite",
        choices=preset_names(),
        help="Built-in HoliTok architecture preset",
    )
    parser.add_argument("--config", default=None, help="Optional custom experiment or model JSON5 config")
    parser.add_argument("--checkpoint", default=None, help="Local checkpoint path")
    parser.add_argument("--repo-id", default=None, help="Optional Hugging Face repo id or checkpoint URL")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision")
    parser.add_argument("--cache-dir", default=None, help="Optional checkpoint cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="Use only cached checkpoint files")
    parser.add_argument("--basic-config", default=None, help="Optional Audiokit base config")
    parser.add_argument("--device", default=None, help="cuda or cuda:0")
    parser.add_argument("--no-sanitize-runtime", action="store_true", help="Keep training-only config branches")


def _load_model(args) -> HoliTok:
    return HoliTok(
        args.config or args.model,
        args.checkpoint,
        basic_config=args.basic_config,
        device=args.device,
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        sanitize_runtime=not args.no_sanitize_runtime,
    )


def encode_cmd(args) -> None:
    vae = _load_model(args)
    audio, _ = load_audio(args.input, vae.sample_rate)
    if args.mode == "posterior":
        latents = vae.encode_posterior(audio)
    else:
        latents = vae.encode(audio, sample=args.mode == "sample", noise_scale=args.noise_scale)
    torch.save(
        {
            "latents": latents.detach().cpu(),
            "mode": args.mode,
            "sample_rate": vae.sample_rate,
            "hop_size": vae.hop_size,
            "latent_dim": vae.latent_dim,
        },
        args.output,
    )


def decode_cmd(args) -> None:
    vae = _load_model(args)
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    latents = payload["latents"] if isinstance(payload, dict) and "latents" in payload else payload
    mode = payload.get("mode", args.mode) if isinstance(payload, dict) else args.mode
    if mode == "posterior" and not args.posterior_do_sample:
        latents = HoliTok.posterior_mean(latents)
        do_sample = False
    else:
        do_sample = mode == "posterior"
    audio = vae.decode(latents, do_sample=do_sample, noise_scale=args.noise_scale)
    save_audio(args.output, audio, vae.sample_rate)


def reconstruct_cmd(args) -> None:
    vae = _load_model(args)
    audio, _ = load_audio(args.input, vae.sample_rate)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    recon = vae.reconstruct(audio, sample=args.mode == "sample", noise_scale=args.noise_scale)
    save_audio(args.output, recon[..., : audio.shape[-1]], vae.sample_rate)


def _load_semantic(args) -> SemanticModule:
    if args.semantic_checkpoint:
        return SemanticModule(args.semantic_checkpoint, device=args.device)
    return SemanticModule.from_pretrained(
        args.model,
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        device=args.device,
    )


def _load_latents_for_semantic(args) -> tuple[torch.Tensor, str, dict]:
    metadata = {}
    if args.input_latents:
        payload = torch.load(args.input_latents, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "latents" in payload:
            latents = payload["latents"]
            metadata.update(payload)
            latent_mode = str(payload.get("mode", args.posterior_mode))
        else:
            latents = payload
            latent_mode = args.posterior_mode

        if latent_mode == "posterior":
            if args.posterior_mode == "sample":
                latents = HoliTok.sample_posterior(latents, noise_scale=args.noise_scale)
                latent_mode = "sample"
            else:
                latents = HoliTok.posterior_mean(latents)
                latent_mode = "mean"
    else:
        vae = _load_model(args)
        audio, _ = load_audio(args.input_audio, vae.sample_rate)
        if args.seed is not None:
            torch.manual_seed(args.seed)
        latents = vae.encode(audio, sample=args.mode == "sample", noise_scale=args.noise_scale)
        latent_mode = args.mode
        metadata.update(
            {
                "sample_rate": vae.sample_rate,
                "hop_size": vae.hop_size,
                "latent_dim": vae.latent_dim,
            }
        )

    if latents.dim() == 2:
        latents = latents.unsqueeze(0)
    if latents.dim() != 3:
        raise ValueError(f"latents must have shape [B,D,T] or [B,T,D], got {tuple(latents.shape)}")
    if not args.channel_last:
        latents = latents.transpose(1, 2)
    return latents, latent_mode, metadata


def semantic_cmd(args) -> None:
    if bool(args.input_audio) == bool(args.input_latents):
        raise ValueError("Pass exactly one of --input-audio or --input-latents")

    latents, latent_mode, metadata = _load_latents_for_semantic(args)
    semantic = _load_semantic(args)
    features = semantic(latents, channel_last=True)
    torch.save(
        {
            "features": features.detach().cpu(),
            "model": args.model,
            "latent_mode": latent_mode,
            "semantic_dim": int(features.shape[-1]),
            "source": args.input_latents or args.input_audio,
            "metadata": metadata,
        },
        args.output,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="holitok")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode = subparsers.add_parser("encode")
    _add_model_args(encode)
    encode.add_argument("--input", required=True)
    encode.add_argument("--output", required=True)
    encode.add_argument("--mode", choices=["posterior", "mean", "sample"], default="posterior")
    encode.add_argument("--noise-scale", type=float, default=1.0)
    encode.set_defaults(func=encode_cmd)

    decode = subparsers.add_parser("decode")
    _add_model_args(decode)
    decode.add_argument("--input", required=True)
    decode.add_argument("--output", required=True)
    decode.add_argument("--mode", choices=["posterior", "mean", "sample"], default="mean")
    decode.add_argument("--posterior-do-sample", action="store_true")
    decode.add_argument("--noise-scale", type=float, default=1.0)
    decode.set_defaults(func=decode_cmd)

    reconstruct = subparsers.add_parser("reconstruct")
    _add_model_args(reconstruct)
    reconstruct.add_argument("--input", required=True)
    reconstruct.add_argument("--output", required=True)
    reconstruct.add_argument("--mode", choices=["mean", "sample"], default="sample")
    reconstruct.add_argument("--noise-scale", type=float, default=1.0)
    reconstruct.add_argument("--seed", type=int, default=None)
    reconstruct.set_defaults(func=reconstruct_cmd)

    semantic = subparsers.add_parser("semantic")
    _add_model_args(semantic)
    semantic.add_argument("--input-audio", default=None)
    semantic.add_argument("--input-latents", default=None)
    semantic.add_argument("--output", required=True)
    semantic.add_argument("--semantic-checkpoint", default=None, help="Optional local semantic.pt path")
    semantic.add_argument("--mode", choices=["mean", "sample"], default="mean")
    semantic.add_argument("--posterior-mode", choices=["mean", "sample"], default="mean")
    semantic.add_argument("--channel-last", action="store_true", help="Input latents are [B,T,D]")
    semantic.add_argument("--noise-scale", type=float, default=1.0)
    semantic.add_argument("--seed", type=int, default=None)
    semantic.set_defaults(func=semantic_cmd)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
