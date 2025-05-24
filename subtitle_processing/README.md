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

- License
MIT License
