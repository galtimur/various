from .get_engine import get_engine_by_name


def dummy_model(prompt: str) -> str:
    return "def reverse_string(s):\n    return ''.join(reversed(s))"


class CodeGenerator:
    """
    Has a method generate(prompt) that returns a string according to prompt.
    """

    def __init__(
        self,
        model_name: str,
        model_pars: dict = {},
        system_prompt: str = "You are a helpful assistant. Return code in ```python block.",
        **kwargs,
    ):
        self.engine = get_engine_by_name(
            model_name, model_pars, system_prompt, **kwargs
        )

    @staticmethod
    def parse_response(response: str) -> str:
        """
        Extract the last Python code block from a response
        """
        # TODO: force the model to include answer tags and extract fromt it
        # Find the last occurrence of ```python
        last_python_block_start = response.rfind("```python")

        if last_python_block_start == -1:
            return ""  # No Python code block found

        # Find the closing ``` after the last ```python
        code_start = last_python_block_start + len("```python")
        code_end = response.find("```", code_start)

        if code_end == -1:
            return ""  # No closing code block

        # Extract the code between the markers
        code = response[code_start:code_end].strip()

        return code

    def generate(self, prompt: str) -> str:
        response = self.engine.make_request(prompt)
        if response is None:
            # TODO think on the error handling
            raise ValueError("Model has not generated anything")
        answer = response.get("response")
        if isinstance(answer, list):
            answer = answer[0]  # for vllm we get lists

        code = self.parse_response(answer)

        return code
