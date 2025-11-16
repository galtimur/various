# %%

from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from generation_engine.litellm_engine import LiteLLMEngine

from dotenv import load_dotenv
import tiktoken
from tqdm import tqdm
import srt

load_dotenv()
# %%


def gen_request(subs, lang_target, lang_res) -> str:

    requests = f"Translate this subtitles srt file from {lang_res} to {lang_target}. ALWAYS return the SAME number of segments. NEVER skip any segment. NEVER combine segments. Return result in srt format. \n\n{subs}"

    return requests


def split_list(input_list, chunk_size=50):
    return [
        input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)
    ]


# %%
if __name__ == "__main__":

    lang_res = "german"
    lang_target = "russian"
    model_name = "openai/gpt-5"
    str_path = "C:/Temp/srt_to_translate.srt"
    str_path_trans = "C:/Temp/srt_to_translate_ru.srt"
    system_prompt = "You are an experienced semantic translator. Follow the instructions carefully."

    kwargs = {
        "model_name": model_name,
        "system_prompt": system_prompt,
        "add_args": {"temperature": 0, "max_tokens": 10000},
        "wait_time": 20.0,
        "attempts": 10,
    }
    model = LiteLLMEngine(**kwargs)

    enc = tiktoken.encoding_for_model("gpt-5")

    # %%

    for i in [1]:  # range(4,9):

        # srt_file_name = str(i)
        print(str_path)
        srt_file = str_path
        srt_file_trans = str_path_trans

        with open(srt_file, "r", encoding="utf-8") as f:
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
            answer_split = answer.split("```srt\n")
            if len(answer_split) > 1:
                parsed_answer = answer.split("```srt\n")[1]
                parsed_answer = parsed_answer.split("```")[0]
            else:
                parsed_answer = answer
            answers.append(parsed_answer)

        # %%

        merged_translated = "\n".join(answers)

        with open(srt_file_trans, "w", encoding="utf-8") as f:
            f.write(merged_translated)

# %%
