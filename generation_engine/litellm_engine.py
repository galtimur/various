import os
import time

import openai


def test_connection(client):
    """Test the connection to the LiteLLM proxy server."""
    try:
        # models = client.models.list()
        # print(f"Connection successful! Available models: {[model.id for model in models.data]}\n")
        return True
    except Exception as e:
        print(f"\nError connecting to proxy server: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your LiteLLM proxy server is running")
        print("2. Check that the base URL is correct")
        print(
            "3. Verify that the model specified is available in your proxy configuration"
        )
        print("4. Run the proxy with debug mode: litellm --proxy --debug")
        return False


class LiteLLMEngine:
    def __init__(
        self,
        model_name: str,
        system_prompt: str = "You are helpful assistant",
        add_args: dict = {},
        wait_time: float = 60.0,
        attempts: int = 10,
    ) -> None:
        api_key = os.getenv("LITELLM_KEY")
        if api_key is None:
            raise ValueError("Please provide LITELLM_KEY in env variables!")

        self.system_prompt = system_prompt
        self.wait_time = wait_time
        self.attempts = attempts
        self.name = model_name

        lite_llm_host = "https://litellm.labs.jb.gg/"
        timeout = 180

        # TODO add parameters
        self.client = openai.OpenAI(
            api_key=api_key, base_url=lite_llm_host) #, timeout=timeout)
        connect_exists = test_connection(self.client)
        if not connect_exists:
            raise ConnectionError("Connection to LiteLLM proxy server failed.")

    def ask(
        self,
        request: str,
    ):

        completion = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": request,
                },
            ],
        )

        return completion.choices[0].message.content

    def make_request(
        self,
        request: str,
    ) -> dict | None:
        error_counts = 0
        response = None

        while error_counts < self.attempts:
            try:
                response = self.ask(request=request)
            except Exception as e:
                print(e)
                print(f"Will wait {self.wait_time} s for another attempt")
                time.sleep(self.wait_time)
                error_counts += 1
            if response is not None:
                return {"response": response}

        return None
