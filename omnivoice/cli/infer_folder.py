"""Batch folder inference CLI for OmniVoice.

Reads all .txt files from an input folder and generates a matching .wav file
for each one in an output folder. The model is loaded once and reused across
all files.

Usage:
    # Auto voice (default folders)
    omnivoice-infer-folder

    # Custom folders
    omnivoice-infer-folder --input_dir texts/ --output_dir audio/

    # Voice cloning applied to all files
    omnivoice-infer-folder --input_dir texts/ --output_dir audio/ \
        --ref_audio ref.wav --ref_text "Reference transcript."

    # Voice design applied to all files
    omnivoice-infer-folder --input_dir texts/ --output_dir audio/ \
        --instruct "male, British accent"
"""

import argparse
import logging
import time
from pathlib import Path

import torch

import soundfile as sf

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.common import str2bool


def get_best_device():
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniVoice batch folder inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="work/input",
        help="Folder containing .txt files to synthesize.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="work/output",
        help="Folder to write output .wav files to.",
    )
    # Voice cloning
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=None,
        help="Reference audio file path for voice cloning.",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default=None,
        help="Reference text describing the reference audio.",
    )
    # Voice design
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Style instruction for voice design mode.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Language name (e.g. 'English') or code (e.g. 'en').",
    )
    # Generation parameters
    parser.add_argument("--num_step", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Fixed output duration in seconds. If set, overrides the "
        "model's duration estimation. The speed factor is automatically "
        "adjusted to match while preserving language-aware pacing.",
    )
    parser.add_argument("--t_shift", type=float, default=0.1)
    parser.add_argument("--denoise", type=str2bool, default=True)
    parser.add_argument(
        "--postprocess_output",
        type=str2bool,
        default=True,
    )
    parser.add_argument("--layer_penalty_factor", type=float, default=5.0)
    parser.add_argument("--position_temperature", type=float, default=5.0)
    parser.add_argument("--class_temperature", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference. Auto-detected if not specified.",
    )
    return parser


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if args.ref_audio:
        output_dir = output_dir / Path(args.ref_audio).stem
    output_dir = output_dir / str(args.speed)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        logging.warning(f"No .txt files found in {input_dir}")
        return

    logging.info(f"Found {len(txt_files)} .txt files in {input_dir}")

    device = args.device or get_best_device()
    logging.info(f"Loading model from {args.model} on {device} ...")
    model = OmniVoice.from_pretrained(
        args.model, device_map=device, dtype=torch.float16
    )

    total_start = time.monotonic()
    skipped = 0
    total_audio_duration = 0.0
    total_chars = 0

    for i, txt_path in enumerate(txt_files, 1):
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            logging.warning(f"Skipping empty file: {txt_path.name}")
            skipped += 1
            continue
        out_path = output_dir / txt_path.with_suffix(".wav").name
        logging.info(f"[{i}/{len(txt_files)}] {txt_path.name} -> {out_path.name}")

        t0 = time.monotonic()
        audios = model.generate(
            text=text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            instruct=args.instruct,
            duration=args.duration,
            num_step=args.num_step,
            guidance_scale=args.guidance_scale,
            speed=args.speed,
            t_shift=args.t_shift,
            denoise=args.denoise,
            postprocess_output=args.postprocess_output,
            layer_penalty_factor=args.layer_penalty_factor,
            position_temperature=args.position_temperature,
            class_temperature=args.class_temperature,
        )
        elapsed = time.monotonic() - t0

        audio = audios[0]
        sf.write(str(out_path), audio, model.sampling_rate)

        audio_duration = audio.shape[-1] / model.sampling_rate
        rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")
        total_audio_duration += audio_duration
        total_chars += len(text)

        logging.info(
            f"  chars={len(text)}  audio={audio_duration:.1f}s  "
            f"infer={elapsed:.1f}s  RTF={rtf:.2f}x"
        )

    total_elapsed = time.monotonic() - total_start
    processed = len(txt_files) - skipped
    logging.info(
        f"Done. {processed} file(s) processed, {skipped} skipped | "
        f"total audio={total_audio_duration:.1f}s  "
        f"total infer={total_elapsed:.1f}s  "
        f"total chars={total_chars}"
    )


if __name__ == "__main__":
    main()
