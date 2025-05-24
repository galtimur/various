# %%

from generation_engine.openai_engine import OpenAIEngine
from dotenv import load_dotenv
import tiktoken
from tqdm import tqdm
import srt

load_dotenv()
# %%


def gen_request(subs, lang_target, lang_res):

    requests = f"Translate this subtitles srt file from {lang_res} to {lang_target}. ALWAYS return the SAME number of segments. NEVER skip any segment. NEVER combine segments. Return result in srt format. \n\n{subs}"

    return requests


def split_list(input_list, chunk_size=50):
    return [
        input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)
    ]


# %%

lang_res = "german"
lang_target = "russian"

kwargs = {
    "model_name": "gpt-4o",
    "system_prompt": "You are an experienced semantic translator. Follow the instructions carefully.",
    "add_args": {"temperature": 0, "max_tokens": 16000},
    "wait_time": 20.0,
    "attempts": 10,
}

model = OpenAIEngine(**kwargs)
enc = tiktoken.encoding_for_model("gpt-4o")

# %%

for i in [9, 10, 11, 12]:  # range(4,9):

    srt_file_name = str(i)
    print(srt_file_name)
    srt_file = f"video/{srt_file_name}.srt"
    srt_file_trans = f"video/{srt_file_name}_ru.srt"

    with open(srt_file, "r") as f:
        subs_txt = f.read()
    if subs_txt[0] == "﻿":
        subs_txt = subs_txt[1:]
    # %%
    subs = list(srt.parse(subs_txt))
    subs_chunks = split_list(subs, chunk_size=50)
    subs_txt_chunks = [srt.compose(chunk, reindex=False) for chunk in subs_chunks]
    request_chunks = [
        gen_request(chunk, lang_target, lang_res) for chunk in subs_txt_chunks
    ]

    tokens = [len(enc.encode(chunk)) for chunk in request_chunks]
    print(f"Number of tokens = {tokens}")

    # %%

    answers = []
    for request in tqdm(request_chunks):
        response = model.make_request(request)
        answer = response["response"]
        parsed_answer = answer.split("```srt\n")[1]
        parsed_answer = parsed_answer.split("```")[0]
        answers.append(parsed_answer)

    # %%

    merged_translated = "\n".join(answers)

    with open(srt_file_trans, "w") as f:
        f.write(merged_translated)

# %%
