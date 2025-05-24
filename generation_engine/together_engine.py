from dotenv import load_dotenv

from .base_engine import BaseOpenAIEngine

load_dotenv()


# TODO make a base class, and reuse it for OpenAIEngine
class TogetherEngine(BaseOpenAIEngine):
    def __init__(
        self,
        model_name,
        system_prompt: str = "You are helpful assistant",
        add_args: dict = {},
        wait_time: float = 20.0,
        attempts: int = 10,
    ) -> None:
        api_key_name = "TOGETHERAI_KEY"
        super().__init__(
            model_name, system_prompt, add_args, wait_time, attempts, api_key_name
        )
        self.model_url = "https://api.together.xyz/v1/chat/completions"
        self.headers.update({"accept": "application/json"})

    @staticmethod
    def get_content(content: list[dict]) -> str:
        content = [cont.get("text") for cont in content if cont.get("text")]  # type: ignore
        return content[0]  # type: ignore
