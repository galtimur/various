#!/usr/bin/env python3
"""
Script to extract audio from video files.
Usage: python extract_audio_script.py --input_folder /path/to/videos --video_ext avi
"""

import argparse
from tqdm import tqdm
from pathlib import Path
import sys
from audio_extract import extract_audio


def extract_audio_from_video_folder(
    input_folder: Path | str,
    output_folder: Path | str,
    video_file_ext: str,
    overwrite: bool = False,
):
    """
    This is a placeholder for your extract_audio function.
    Import your actual implementation from audio_extract.py
    """

    video_files = list(input_folder.glob(f"*.{video_file_ext}"))
    if not video_files:
        print(f"No files with extension '{video_file_ext}' found in {input_folder}")
        sys.exit(0)
    print(f"Found {len(video_files)} video files to process.")

    for video_file in tqdm(video_files, desc="Extracting audio"):
        audio_file = output_folder / video_file.with_suffix(".mp3").name

        # Skip if file exists and overwrite is not enabled
        if audio_file.exists() and not overwrite:
            print(
                f"Skipping {video_file.name} - {audio_file.name} already exists. Use --overwrite to force conversion."
            )
            continue

        try:
            extract_audio(
                input_path=str(video_file),
                output_path=str(audio_file),
                overwrite=overwrite,
            )
        except Exception as e:
            print(f"Error processing {video_file.name}: {str(e)}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract audio from video files.")
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to folder containing video files",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Path to output folder for audio files (defaults to input folder)",
    )
    parser.add_argument(
        "--video_ext",
        type=str,
        default="avi",
        help="Video file extension (e.g., avi, mp4, mov)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing audio files"
    )

    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    if not input_folder.exists() or not input_folder.is_dir():
        print(
            f"Error: Input folder '{input_folder}' does not exist or is not a directory."
        )
        sys.exit(1)

    output_folder = Path(args.output_folder) if args.output_folder else input_folder
    if not output_folder.exists():
        print(f"Creating output folder: {output_folder}")
        output_folder.mkdir(parents=True, exist_ok=True)

    extract_audio_from_video_folder(
        input_folder, output_folder, args.video_ext, args.overwrite
    )


if __name__ == "__main__":
    main()
