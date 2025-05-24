#%%
from tqdm import tqdm
from pathlib import Path

from audio_extract import extract_audio


input_folder = Path("C:/Timur/Films/Sankt Maik/S2")
output_folder = input_folder
video_file_ext = "avi"

video_files = list(Path(input_folder).absolute().glob(f'*.{video_file_ext}'))

#%%
# Convert video to mp3
for video_file in tqdm(video_files):

    audio_file = video_file.with_suffix('.mp3')
    extract_audio(input_path=str(video_file),
                  output_path=str(audio_file),
                  overwrite=True)
