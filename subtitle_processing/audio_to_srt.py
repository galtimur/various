#!/usr/bin/env python3
"""
Script to generate subtitles for multiple MP3 files using Whisper.
Usage: python generate_subtitles.py --input_folder /path/to/mp3s --output_folder /path/to/output
"""

import argparse
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import logging
import concurrent.futures
import srt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler("subtitle_generation.log"),
        logging.StreamHandler()
    ],
)

def clean_subtitles(sub_folder: str | Path, output_dir: str | Path = None):
    """Clean up subtitles text in the folder"""

    with open("hallucinations_ru.txt", "r", encoding="utf-8") as f:
        halluc = f.read().splitlines()

    sub_files = list(sub_folder.glob("*.srt"))
    num_hal = 0
    for srt_file in sub_files:
        with open(srt_file, "r", encoding="utf-8") as f:
            subs_txt = f.read()
        subs = srt.parse(subs_txt)
        subs_clean = []
        for sub in subs:
            if not (sub.content in halluc):
                subs_clean.append(sub)
            else:
                num_hal += 1
        subs_clean = srt.sort_and_reindex(subs_clean)
        subs_clean = srt.compose(subs_clean, reindex=False)
        with open(output_dir / srt_file.name, "w", encoding="utf-8") as f:
            f.write(subs_clean)
    print(f'Number of hallucinations = {num_hal}')


def generate_subtitle(mp3_file: str | Path, output_dir: str | Path, args):
    """Generate subtitles for a single MP3 file using Whisper."""
    cmd = [
        "CUDA_VISIBLE_DEVICES=" + str(args.gpu_id),
        "whisper", f'"{str(mp3_file)}"',
        "--output_dir", str(output_dir),
        "--model", args.model,
        "--output_format", "srt",
        "--word_timestamps", "True",
        "--max_line_count", args.max_line_count,
        "--max_words_per_line", args.max_words_per_line,
        "--verbose", "False",
    ]
    if args.language is not None:
        cmd.extend(["--language", args.language])
    try:
        # Run the command
        logging.info(f"Generating subtitles for {mp3_file.name}")
        process = subprocess.run(" ".join(cmd), shell=True, check=True, text=True)

        logging.info(f"Successfully generated subtitles for {mp3_file.name}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error processing {mp3_file.name}: {e}")
        logging.error(f"Command output: {e.stdout}")
        logging.error(f"Command error: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error with {mp3_file.name}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate subtitles for multiple MP3 files."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to folder containing MP3 files",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Path to output folder for subtitle files (defaults to input folder)",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="GPU ID to use for processing"
    )
    parser.add_argument(
        "--max_line_count", type=str, default="2", help="Maximal lines in single sub"
    )
    parser.add_argument(
        "--max_words_per_line",
        type=str,
        default="15",
        help="Maximal words in single sub",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="turbo",
        help="Whisper model to use (e.g., tiny, small, medium, large, turbo)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., en, ru, fr, de ..)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process files in parallel (may require multiple GPUs)",
    )
    parser.add_argument(
        "--clean",
        dest="clean",
        action="store_const",
        const=True,
        default=True,
        help="Enable cleaning the subtitles from hallucinations (default: True)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers when --parallel is used",
    )

    args = parser.parse_args()

    # Set up input/output paths
    input_folder = Path(args.input_folder)
    if not input_folder.exists() or not input_folder.is_dir():
        logging.error(
            f"Input folder '{input_folder}' does not exist or is not a directory."
        )
        sys.exit(1)

    output_folder = Path(args.output_folder) if args.output_folder else input_folder
    if not output_folder.exists():
        logging.info(f"Creating output folder: {output_folder}")
        output_folder.mkdir(parents=True, exist_ok=True)

    # Get all MP3 files
    mp3_files = list(input_folder.glob("*.mp3"))
    if not mp3_files:
        logging.warning(f"No MP3 files found in {input_folder}")
        sys.exit(0)

    logging.info(f"Found {len(mp3_files)} MP3 files to process")

    # TODO make possible parallel execution with free GPU selection
    # if args.parallel and args.max_workers > 1:
    #     logging.info(f"Processing files in parallel with {args.max_workers} workers")
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    #         futures = []
    #         for mp3_file in mp3_files:
    #             future = executor.submit(
    #                 generate_subtitle,
    #                 mp3_file,
    #                 output_folder,
    #                 args.gpu_id,
    #                 args.model,
    #                 args.language
    #             )
    #             futures.append(future)
    #
    #         # Show progress bar
    #         for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating subtitles"):
    #             pass

    for mp3_file in tqdm(mp3_files, desc="Generating subtitles"):
        generate_subtitle(mp3_file, output_folder, args)

    clean_subtitles(output_folder, output_folder / "clean_subs")

    logging.info("Subtitle generation complete")


if __name__ == "__main__":
    main()
