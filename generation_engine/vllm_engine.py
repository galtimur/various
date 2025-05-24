import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vllm import LLM, RequestOutput, SamplingParams


def check_files_exist(folder_path: Path | str, filenames: list[str]) -> bool:
    folder_path = Path(folder_path)
    existing_files = [(folder_path / filename).exists() for filename in filenames]
    return all(existing_files)


def get_model_name_and_path(
    model_name_or_path: str | Path, model_name: str | None = None
) -> tuple[str, str | None]:
    tokenizer_files = [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
    ]
    tok_files_exist = check_files_exist(model_name_or_path, tokenizer_files)
    tok_name_or_path = None
    if not Path(model_name_or_path).exists():
        model_name = str(model_name_or_path)
    else:
        config_path = Path(model_name_or_path) / "config.json"
        if config_path.exists():
            with open(config_path, "r") as file:
                config_data = json.load(file)
            if config_data.get("_name_or_path") is not None:
                model_name = config_data.get("_name_or_path")
        if (not tok_files_exist) and (model_name is None):
            raise AttributeError(
                "You have no tokenizer files in model folder.\n"
                "Please provide model name for tokenizer either in config.json file in the model folder\n"
                "or as model.model_name parameter"
            )
        elif tok_files_exist and model_name is None:
            model_name = str(model_name_or_path)
        else:
            tok_name_or_path = model_name

    return model_name, tok_name_or_path  # type: ignore


class VllmEngine:
    def __init__(
        self,
        model_name: str,
        system_prompt: str = "You are helpful assistant",
        add_args: dict = {},
        vllm_args: dict = {},
        generation_args: dict = {},
    ):
        self.name, tokenizer_name = get_model_name_and_path(
            model_name_or_path=model_name
        )
        if tokenizer_name is not None:
            vllm_args["tokenizer"] = tokenizer_name

        # Set max model length if not provided
        if "max_model_len" not in vllm_args:
            vllm_args["max_model_len"] = 32000  # Slightly below the max to be safe

        if "temperature" in add_args:
            generation_args.update({"temperature": add_args["temperature"]})
        else:
            generation_args.update({"temperature": 0.0})

        if "max_tokens" not in generation_args:
            generation_args.update(
                {"max_tokens": add_args["max_tokens"], "ignore_eos": False}
            )
        else:
            # some deafult value here
            generation_args.update({"max_tokens": 1024, "ignore_eos": False})
        self.llm = LLM(model=model_name, **vllm_args)
        self.sampling_params = SamplingParams(**generation_args)
        self.system_prompt = system_prompt

        # Store tokenizer for length checking
        self.tokenizer = self.llm.get_tokenizer()
        self.max_input_length = vllm_args.get(
            "max_model_len", 32000
        ) - generation_args.get("max_tokens", 1024)

    def generate(
        self,
        input_texts: list[str],
    ) -> dict[str, list[Any]]:
        formatted_texts = [self.format_input(text) for text in input_texts]
        # Check and truncate inputs if needed
        truncated_inputs = []
        for text in formatted_texts:
            token_count = len(self.tokenizer.encode(text))
            if token_count > self.max_input_length:
                print(
                    f"Warning: Input length ({token_count} tokens) exceeds maximum allowed ({self.max_input_length}). Truncating."
                )
                # Truncate by reducing the message content
                truncated_text = self.truncate_input(text, self.max_input_length)
                truncated_inputs.append(truncated_text)
            else:
                truncated_inputs.append(text)

        responses = self.llm.generate(
            prompts=truncated_inputs,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        outputs = [self.get_outputs(response) for response in responses]

        return self.batch_output(outputs)

    def truncate_input(self, text: str, max_length: int) -> str:
        """Truncate input text to fit within token limit."""
        # Simple approach: For chat formats, preserve system prompt and truncate user message
        if "meta-llama" in self.name or "qwen" in self.name.lower():
            # Split by markers to identify parts
            if "meta-llama" in self.name:
                parts = text.split("<|eot_id|>")
                if len(parts) >= 2:  # We have system and user parts
                    system_part = parts[0] + "<|eot_id|>"
                    user_part = parts[1]
                    assistant_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"

                    # Calculate how many tokens we can keep from user message
                    system_tokens = len(self.tokenizer.encode(system_part))
                    assistant_tokens = len(self.tokenizer.encode(assistant_part))
                    available_tokens = max_length - system_tokens - assistant_tokens

                    # Truncate user message
                    user_tokens = self.tokenizer.encode(user_part)
                    if len(user_tokens) > available_tokens:
                        truncated_user = self.tokenizer.decode(
                            user_tokens[:available_tokens]
                        )
                        return system_part + truncated_user + assistant_part

            elif "qwen" in self.name.lower():
                parts = text.split("<|im_end|>\n")
                if len(parts) >= 2:  # We have system and user parts
                    system_part = parts[0] + "<|im_end|>\n"
                    user_part = parts[1]
                    assistant_part = "<|im_start|>assistant\n<|im_start|>think\n"

                    # Calculate how many tokens we can keep from user message
                    system_tokens = len(self.tokenizer.encode(system_part))
                    assistant_tokens = len(self.tokenizer.encode(assistant_part))
                    available_tokens = max_length - system_tokens - assistant_tokens

                    # Truncate user message
                    user_tokens = self.tokenizer.encode(user_part)
                    if len(user_tokens) > available_tokens:
                        truncated_user = self.tokenizer.decode(
                            user_tokens[:available_tokens]
                        )
                        return system_part + truncated_user + assistant_part

        # Fallback: simple truncation based on tokens
        tokens = self.tokenizer.encode(text)
        if len(tokens) > max_length:
            return self.tokenizer.decode(tokens[:max_length])
        return text

    def format_input(self, message: str) -> str:
        if "meta-llama" in self.name:  # TODO: haven't tested this one
            system_mes = f"<|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}<|eot_id|>"
            user_mes = (
                f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>"
            )
            assist_prompt = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            model_input = system_mes + user_mes + assist_prompt
            # else:
        #     model_input = (self.system_prompt + "\n" + message).strip()
        elif "qwen" in self.name.lower():
            system_mes = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            user_mes = f"<|im_start|>user\n{message}<|im_end|>\n"
            assist_prompt = "<|im_start|>assistant\n<|im_start|>think\n"  # prefilling think to force model to reason
            model_input = system_mes + user_mes + assist_prompt
        else:
            model_input = message
        return model_input

    def make_request(
        self,
        request: str | list[str],
    ) -> dict | None:
        if isinstance(request, str):
            requests = [request]
        else:
            requests = request
        requests = [self.format_input(request) for request in requests]
        response = self.generate(input_texts=requests)

        return {"response": response["text"]}

    @staticmethod
    def get_outputs(response: RequestOutput) -> dict[str, Any]:
        metainfo = asdict(response.outputs[0])  # type: ignore
        del metainfo["text"], metainfo["token_ids"]
        metainfo["time_metrics"] = response.metrics  # type: ignore
        output_dict = {
            "text": response.outputs[0].text,
            "tokens": list(response.outputs[0].token_ids),
            "metainfo": metainfo,
        }
        return output_dict

    @staticmethod
    def batch_output(outputs: list[dict[str, Any]]) -> dict[str, list[Any]]:
        batched_output = defaultdict(list)
        for d in outputs:
            for key, value in d.items():
                batched_output[key].append(value)
        return dict(batched_output)
