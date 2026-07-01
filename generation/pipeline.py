import logging
from typing import List
from retrieval.schema import FinalContext
from generation.schema import GenerationResult
from generation.context_builder import ContextBuilder
from generation.prompt_builder import PromptBuilder
from generation.llm import LLMProvider, GeminiProvider

logger = logging.getLogger(__name__)

class GenerationPipeline:
    """
    Orchestrates the final RAG generation step:
    FinalContexts -> Context Builder -> Prompt Builder -> LLM -> Output
    """
    def __init__(self, provider: LLMProvider = None):
        self.context_builder = ContextBuilder()
        self.prompt_builder = PromptBuilder()
        
        # Default to Gemini if not provided
        self.llm = provider or GeminiProvider()
        
    def generate(self, query: str, contexts: List[FinalContext]) -> GenerationResult:
        if not contexts:
            logger.warning("No contexts provided for generation.")
            return GenerationResult(
                answer="I cannot answer this question as no relevant documents were found.",
                sources=[],
                model_used="none",
                usage_metadata={}
            )
            
        logger.info("Building context from %d retrieved items...", len(contexts))
        context_str, sources = self.context_builder.build(contexts)
        
        logger.info("Building prompt...")
        system_prompt = self.prompt_builder.build_system_prompt(context_str)
        user_prompt = self.prompt_builder.build_user_prompt(query)
        
        logger.info("Calling LLM provider...")
        answer, model_used, usage = self.llm.generate(system_prompt, user_prompt)
        
        logger.info("Generation complete (Model: %s).", model_used)
        return GenerationResult(
            answer=answer,
            sources=sources,
            model_used=model_used,
            usage_metadata=usage
        )
