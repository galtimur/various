from dotenv import load_dotenv

# moved imports to avoid unnecessary long imports, especially VLLM, if it is not needed.

load_dotenv()


def get_engine_by_name(
    model_name: str,
    model_pars: dict = {},
    system_prompt: str = "",
    **kwargs,
):
    kwargs.update(
        {
            "add_args": model_pars,
            "model_name": model_name,
        }
    )
    if system_prompt:
        kwargs.update({"system_prompt": system_prompt})

    if model_name.startswith("openai/"):
        from .openai_engine import OpenAIEngine

        kwargs.update({"model_name": model_name[len("openai/") :]})
        engine = OpenAIEngine(**kwargs)  # type: ignore
    elif model_name.startswith("litellm/"):
        from .litellm_engine import LiteLLMEngine

        kwargs.update({"model_name": model_name[len("litellm/") :]})
        engine = LiteLLMEngine(**kwargs)  # type: ignore
    elif model_name.startswith("grazie/"):
        from .grazie_engine import Grazie

        kwargs.update({"model_name": model_name[len("grazie/") :]})
        engine = Grazie(**kwargs)  # type: ignore
    elif model_name.startswith("together/"):
        from .together_engine import TogetherEngine

        kwargs.update({"model_name": model_name[len("together/") :]})
        engine = TogetherEngine(**kwargs)  # type: ignore
    else:
        from .vllm_engine import VllmEngine

        engine = VllmEngine(**kwargs)  # type: ignore

    return engine
