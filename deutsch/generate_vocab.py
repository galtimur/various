# %% md
# Batch translation pipeline: invoke LLM, parse JSON lines, and append outputs
# %%

import os
import re
import json
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# %%
# Output paths
RAW_RESPONSES_PATH = Path("raw_responses.jsonl")
PARSED_JSONL_PATH = Path("vocab_cards_gpt5.jsonl")


def get_words_batches(word_list_file, batch_size):
    with open(word_list_file, "r", encoding="utf-8") as f:
        word_list = [line.strip() for line in f if line.strip()]
    word_list = list(set([re.sub(r" {2,}", " ", word) for word in word_list]))
    words_batches = [word_list[i:i + batch_size] for i in range(0, len(word_list), batch_size)]

    return words_batches

def get_messages(words_batches, system_prompt, base_prompt):

    messages = []
    for batch in words_batches:
        batch_str = "\n".join(batch)
        prompt = (base_prompt + batch_str).strip()
        message = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        messages.append(message)

    return messages

def ensure_files():
    RAW_RESPONSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARSED_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Touch files if not exist
    RAW_RESPONSES_PATH.touch(exist_ok=True)
    PARSED_JSONL_PATH.touch(exist_ok=True)


def extract_translation_block(text: str) -> str:
    """
    Extracts the block between the indicator after TRANSLATION_LIST``` and before the closing ``` (if present).
    If indicators are missing, returns the whole text as a fallback.
    """
    if not text:
        return ""
    # Try to find the indicator after TRANSLATION_LIST```
    start_marker = "TRANSLATION_LIST```"
    end_marker = "```"
    start_idx = text.find(start_marker)
    if start_idx != -1:
        start_idx = start_idx + len(start_marker)
        # If there's another code fence after, cut there
        end_idx = text.find(end_marker, start_idx)
        if end_idx != -1:
            return text[start_idx:end_idx].strip()
        return text[start_idx:].strip()
    return text.strip()


def parse_json_lines(block: str) -> tuple[list[dict], list[str]]:
    """
    Parse a block that should contain one JSON object per line.
    Returns (parsed_items, raw_lines_kept).
    Lines that fail to parse are ignored.
    """
    items = []
    kept_raw = []
    for line in block.splitlines():
        s = line.strip()
        if not s:
            continue
        # Heuristic: keep only lines that look like JSON objects
        if not (s.startswith("{") and s.endswith("}")):
            continue
        try:
            obj = json.loads(s)
            # Basic shape validation
            if isinstance(obj, dict) and "word" in obj and "translation" in obj:
                items.append(obj)
                kept_raw.append(s)
        except json.JSONDecodeError:
            continue
    return items, kept_raw


def append_lines(path: Path, lines: Iterable[str]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def main():

    word_list_file = "C:/Timur/GoogleDrive/Timur/Deutsch/words_B1.txt"

    address = "https://litellm.labs.jb.gg/"
    api_key = os.getenv("LITELLM_KEY")
    model_name = "gpt-5"
    batch_size = 10
    batch_size_api = 16  # safety chunk for llm.batch if needed
    system_prompt = "You are a helpful assistant that knows both German and Russian. Helpful translator"
    base_prompt = '''
    You're given a list of German words or short phrases, separated by a new line after the indicator WORDS_LIST. For each word generate a json {"word": word, "translation": translation of the word to Russian, "example": example of the word usage, "example_translation": translation of the example to russian}. Generated examples should not be templated; they should be natural phrases taken from the natural texts. Each JSON should start on a new line, forming a JSONL file. Start list of JSON dicts with indicator TRANSLATION_LIST```, then list translations of words in the list, end with indicator ```

    WORDS_LIST:
    '''

    llm = ChatOpenAI(base_url=address, api_key=api_key, model=model_name)

    words_batches = get_words_batches(word_list_file, batch_size)
    messages_all = get_messages(words_batches, system_prompt, base_prompt)


    # Main loop: iterate over all messages, invoke LLM, save raw, parse and save parsed
    ensure_files()

    for chunk in tqdm(chunked(messages_all, batch_size_api), total=len(messages_all) // batch_size_api + 1):
        # Invoke LLM on a chunk of messages
        responses = llm.batch(chunk)
        # responses is a list aligning with chunk
        for resp in responses:
            # 1) Save raw response text
            text = getattr(resp, "content", "") if resp is not None else ""
            if not text:
                continue
            append_lines(RAW_RESPONSES_PATH, [text])
            with open(RAW_RESPONSES_PATH, "a", encoding="utf-8") as f:
                f.write("\n" + 40 * "-" + "\n")

            # 2) Extract the JSONL block and parse
            block = extract_translation_block(text)
            parsed_items, kept_raw = parse_json_lines(block)

            # 3) Append parsed JSON objects to parsed file (one per line)
            if parsed_items:
                append_lines(PARSED_JSONL_PATH, [json.dumps(obj, ensure_ascii=False) for obj in parsed_items])

if __name__ == "__main__":
    main()