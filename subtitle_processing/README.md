# Subtitle generation

Steps:

- Extract audio (mp3) from video. See [Video Audio Extractor](##Video Audio Extractor) section
- Generate str from mp3. See [Subtitles generation](##Subtitles generation) section

## Video Audio Extractor
A simple utility script for extracting audio tracks from video files in batch mode.
This script provides an easy way to extract audio (in MP3 format) from multiple video files stored in a directory. It uses FFmpeg in the background to handle the audio extraction process.

### Basic usage:

`python extract_audio_script.py --input_folder /path/to/videos --video_ext avi`

#### Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--input_folder` | Path to folder containing video files | *Required* |
| `--output_folder` | Path to output folder for audio files | Same as input folder |
| `--video_ext` | Video file extension (e.g., avi, mp4, mov) | avi |
| `--overwrite` | Flag to overwrite existing audio files | False |

#### Examples
Extract audio from all AVI files in a folder:
```bash
python extract_audio_script.py --input_folder /path/to/videos
```

Extract audio from MP4 files to a different output folder:
```bash
python extract_audio_script.py --input_folder /path/to/videos --output_folder /path/to/audio --video_ext mp4
```

Force overwriting existing audio files:
```bash
python extract_audio_script.py --input_folder /path/to/videos --overwrite
```
## How It Works

1. The script scans the specified input directory for video files with the given extension
2. For each video file, it creates an MP3 audio file with the same name
3. By default, it skips existing files unless the `--overwrite` flag is used
4. A progress bar shows the extraction progress

## Notes

- The audio files will be named the same as the video files but with a `.mp3` extension
- If the output directory doesn't exist, it will be created automatically
- The script requires the custom `audio_extract.py` module which handles the actual audio extraction process

## Troubleshooting

- Ensure FFmpeg is properly installed and accessible from your command line
- Check that the `audio_extract.py` module is available
- For permissions issues when creating output directories, try running with elevated privileges

## Subtitles generation 

A utility script to generate SRT subtitles for multiple MP3 files using OpenAI's Whisper speech recognition model.

### Overview

This script automates the process of creating subtitles for audio files by:
- Processing all MP3 files in a specified directory
- Using Whisper's speech recognition capabilities
- Generating subtitle files in SRT format with word-level timestamps
- Supporting custom formatting options for subtitles

### Usage

Basic usage:

```bash
python generate_subtitles.py --input_folder /path/to/mp3s
```

#### Arguments

| Argument               | Description                                              | Default              |
|------------------------|----------------------------------------------------------|----------------------|
| `--input_folder`       | Path to folder containing MP3 files                      | *Required*           |
| `--output_folder`      | Path to output folder for subtitle files                 | Same as input folder |
| `--clean`              | To clean the subs from hallucinations                    | True                 |
| `--gpu_id`             | GPU ID to use for processing                             | 0                    |
| `--max_line_count`     | Maximum lines in a single subtitle                       | 2                    |
| `--max_words_per_line` | Maximum words in a single subtitle line                  | 15                   |
| `--model`              | Whisper model to use (tiny, small, medium, large, turbo) | turbo                |
| `--language`           | Language code (e.g., en, ru, fr, de)                     | auto-detect          |
| `--parallel`           | Flag to process files in parallel (Not used now)         | False                |
| `--max_workers`        | Maximum number of parallel workers (Not used now)        | 1                    |

### Examples

Generate Russian subtitles for all MP3 files in a folder:
```bash
python generate_subtitles.py --input_folder /path/to/mp3s --language ru
```

Use a specific model and GPU:
```bash
python generate_subtitles.py --input_folder /path/to/mp3s --model medium --gpu_id 1
```

Customize subtitle formatting:
```bash
python generate_subtitles.py --input_folder /path/to/mp3s --max_line_count 3 --max_words_per_line 10
```

### Output

The script generates SRT subtitle files with the same base name as the input MP3 files. For example:
- Input: `/path/to/mp3s/lecture.mp3`
- Output: `/path/to/mp3s/lecture.srt` (or in the specified output folder)

License
MIT License
