import logging

logger = logging.getLogger(__name__)

class PromptBuilder:
    """
    Constructs the grounded RAG prompts.
    """
    
    def build_system_prompt(self, context_str: str) -> str:
        """
        Builds the strict system instruction prompting the LLM.
        """
        prompt = (
            "You are an expert Q&A assistant for a Knowledge Base system. "
            "You will be provided with a user query and a set of retrieved documents. "
            "Your task is to answer the user's query based strictly on the provided documents.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. You MUST rely ONLY on the provided documents to form your answer. "
            "Do not use outside knowledge.\n"
            "2. If the provided documents do not contain the answer, you must clearly state: "
            "\"I cannot answer this question based on the provided documents.\"\n"
            "3. Cite the Document number in your response if you use its information.\n"
            "4. Be concise, clear, and professional.\n\n"
            "=== RETRIEVED DOCUMENTS ===\n"
            f"{context_str}\n"
            "===========================\n"
        )
        return prompt

    def build_user_prompt(self, query: str) -> str:
        """
        Builds the user portion of the prompt.
        """
        return f"User Query: {query}"
