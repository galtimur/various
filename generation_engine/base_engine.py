import os.path
import re
import time
from copy import deepcopy
from typing import Any

import requests


class BaseOpenAIEngine:
    def __init__(
        self,
        model_name,
        system_prompt: str = "You are helpful assistant",
        add_args: dict = {},
        wait_time: float = 20.0,
        attempts: int = 10,
        api_key_name: str = "OPENAI_KEY",
    ) -> None:
        api_key = os.getenv(api_key_name)
        if api_key is None:
            raise ValueError(f"Please provide {api_key_name} in env variables!")
        self.headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
        }
        self.name = model_name
        self.system_prompt = system_prompt
        self.wait_time = wait_time
        self.attempts = attempts
        self.args = add_args
        self.payload = {
            "model": model_name,
            "messages": [{"role": "system", "content": system_prompt}],
            "max_tokens": 3000,
        }
        self.payload.update(add_args)
        self.model_url = ""

    @staticmethod
    def get_content(content: list[dict]) -> list[dict] | str:
        return content

    def ask(
        self,
        request: str,
    ) -> dict[str, Any] | None:
        content = [{"type": "text", "text": request}]

        payload = deepcopy(self.payload)
        payload["messages"].append(
            {"role": "user", "content": self.get_content(content)}
        )
        assert self.model_url, "Set model_url first"
        response = requests.post(self.model_url, headers=self.headers, json=payload)

        try:
            return response.json()
        except requests.exceptions.JSONDecodeError as e:
            print(e)
            print(f"GPT response is not json-parsable. Response:\n{response}")
            return None

    def make_request(
        self,
        request: str,
    ) -> dict | None:

        error_counts = 0
        response = None

        while error_counts < self.attempts:
            response = self.ask(request=request)

            if response is None:
                return None

            if "error" not in response.keys():
                response["response"] = response["choices"][0]["message"]["content"]
                response["choices"][0]["message"]["content"] = "MOVED to response key"
                break
            else:
                error_counts += 1
                message = response["error"]["message"]
                seconds_to_wait = re.search(r"Please try again in (\d+)s\.", message)
                if seconds_to_wait is not None:
                    wait_time = 1.5 * int(seconds_to_wait.group(1))
                    print(f"Waiting {wait_time} s")
                    time.sleep(wait_time)
                else:
                    print(
                        f"Cannot parse retry time from error message. Will wait for {self.wait_time} seconds"
                    )
                    print(message)
                    response = None
                    time.sleep(self.wait_time)

        if (response is not None) and ("response" not in response):
            response = None

        return response
