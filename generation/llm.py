import os
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from generation import config

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Generate an answer.
        Returns:
            (answer_text, model_used, usage_metadata)
        """
        pass

class GeminiProvider(LLMProvider):
    def __init__(self):
        try:
            from google import genai
            self.genai = genai
        except ImportError:
            raise ImportError("google-genai package is not installed. Please install it.")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment.")
            
        self.client = self.genai.Client(api_key=api_key)
        self.preferred_model = config.LLM_MODEL
        self.fallback_models = config.GEMINI_FALLBACK_MODELS
        
    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, str, Dict[str, Any]]:
        models_to_try = [self.preferred_model]
        for m in self.fallback_models:
            if m not in models_to_try:
                models_to_try.append(m)
                
        last_error = None
        for model in models_to_try:
            logger.info("Attempting generation with model: %s", model)
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=self.genai.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=config.TEMPERATURE,
                        max_output_tokens=config.MAX_OUTPUT_TOKENS,
                    )
                )
                
                # Extract usage metadata
                usage = {}
                if response.usage_metadata:
                    usage = {
                        "prompt_tokens": response.usage_metadata.prompt_token_count,
                        "candidates_tokens": response.usage_metadata.candidates_token_count,
                        "total_tokens": response.usage_metadata.total_token_count
                    }
                    
                return response.text, model, usage
                
            except Exception as e:
                logger.warning("Model %s failed: %s", model, e)
                last_error = e
                # Continue to next fallback model
                continue
                
        logger.error("All Gemini models failed. Last error: %s", last_error)
        raise RuntimeError(f"LLM Generation failed after trying models {models_to_try}") from last_error

# Placeholders for future providers
class OpenAIProvider(LLMProvider):
    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, str, Dict[str, Any]]:
        raise NotImplementedError("OpenAI provider not yet implemented.")

class LocalProvider(LLMProvider):
    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, str, Dict[str, Any]]:
        raise NotImplementedError("Local provider not yet implemented.")
