from crewai import LLM
import os
from dotenv import load_dotenv

load_dotenv()


def get_advanced_reasoning_model(model = os.getenv("OPENAI_MODEL_NAME")):
    return LLM(
        model=model,
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        reasoning_effort="high",
        max_tokens=None,
    )


def get_high_reasoning_model(
    model=os.getenv("GEMINI_MODEL_NAME"),
    base_url=os.getenv("GEMINI_API_BASE"),
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.6):
    return LLM(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        reasoning_effort="high",
    )


def get_medium_reasoning_model(
    model=os.getenv("GEMINI_MODEL_NAME"),
    base_url=os.getenv("GEMINI_API_BASE"),
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.6):
    return LLM(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        reasoning_effort="medium",
    )


def get_low_reasoning_model(
    model=os.getenv("GEMINI_MODEL_NAME"),
    base_url=os.getenv("GEMINI_API_BASE"),
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.6):
    return LLM(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )
