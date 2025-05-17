#%%
import os
from pathlib import Path

import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from whisper.utils import get_writer
from audio_extract import extract_audio

is_flash_attn_2_available()

#%%

def convert_results(result):
    result["segments"] = result["chunks"]

    for segment in result["segments"]:
        segment["start"] = segment["timestamp"][0]
        segment["end"] = segment["timestamp"][1]
        # del segment["timestamp"]
    
    return result

#%%

# output_directory = "srt_to_transl"
input_folder = ""
output_folder = input_folder

#%%

model_kwargs= {"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"}

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
    torch_dtype=torch.float16,
    device="cuda:0", # or mps for Mac devices
    # model_kwargs=model_kwargs,
)
srt_writer = get_writer("srt", output_folder)
#%%

files = os.listdir(input_folder)
#%%
# Convert video to mp3
for file_name in files[0:1]:

    input_filepath = Path(input_folder) / file_name

    if input_filepath.suffix == '.avi':
        audio_path = str(input_filepath.with_suffix('.mp3'))
        extract_audio(input_path=str(input_filepath),
                      output_path=audio_path,
                      overwrite=True)
    else:
        audio_path = input_filepath

#%%

batch_size = 24
for file_name in files:

    input_filepath = Path(input_folder) / file_name
    if not input_filepath.suffix == '.mp3':
        continue

    result = pipe(
        str(input_filepath),
        chunk_length_s=5,
        batch_size=batch_size,
        return_timestamps=True,
        generate_kwargs = {"task": "transcribe", "language": "ru"}
    )

    result = convert_results(result)

    srt_writer(result, file_name)
    print(f"Subs has been extracted to {output_folder}")

#%%

