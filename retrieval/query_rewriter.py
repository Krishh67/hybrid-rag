import os
import json
import logging
from typing import List
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class QueryRewriter:
    def __init__(self):
        load_dotenv()
        try:
            from google import genai
            self.genai = genai
        except ImportError:
            self.genai = None
            
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or not self.genai:
            logger.warning("GEMINI_API_KEY not found or google-genai missing. QueryRewriter will only return original query.")
            self.client_ready = False
            self.client = None
        else:
            self.client_ready = True
            self.client = None
            
        self.fallback_models = [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-3-flash"
        ]
        
    def _parse_response(self, response_text: str, original_query: str) -> List[str]:
        """Parses the LLM response into a list of queries."""
        try:
            # Try to parse as JSON first
            # We strip markdown formatting if any
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()
            
            reformulations = json.loads(clean_text)
            if isinstance(reformulations, list):
                # Ensure original is first and max 3 reformulations
                results = [original_query]
                for r in reformulations:
                    if isinstance(r, str) and r.strip() and r.strip() != original_query:
                        results.append(r.strip())
                return results[:4]
        except Exception:
            pass
            
        # Fallback to line-by-line parsing
        results = [original_query]
        if response_text:
            for line in response_text.split('\n'):
                cleaned = line.strip()
                # Remove bullets or numbers
                if cleaned.startswith("- ") or cleaned.startswith("* "):
                    cleaned = cleaned[2:].strip()
                elif len(cleaned) > 2 and cleaned[0].isdigit() and cleaned[1] in (".", ")"):
                    cleaned = cleaned[2:].strip()
                    
                if cleaned and cleaned != original_query:
                    results.append(cleaned)
                    if len(results) == 4:
                        break
                    
        return results

    def rewrite(self, query: str) -> List[str]:
        """
        Reformulates the query into up to 3 semantically different alternatives.
        Returns a list where index 0 is always the original query.
        """
        if not self.client_ready:
            return [query]
            
        if self.client is None:
            self.client = self.genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            
        prompt = f"""You are an expert search query reformulator.
The user provided the following search query: "{query}"

Your task is to generate up to 0-3 semantically different reformulations of this query to maximize search recall. 
Do NOT change the underlying intent of the user's query. Use different vocabulary or phrasing (e.g. methodologies instead of methods).
Return ONLY a JSON array of strings containing the reformulations. Do not include the original query. Do not include any other text.
"""

        for model_name in self.fallback_models:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=self.genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.3
                    )
                )
                
                results = self._parse_response(response.text, query)
                logger.info("Successfully rewrote query using %s. Total queries: %d", model_name, len(results))
                print("Rewritten queries: ", results)
                return results
                
            except Exception as e:
                logger.warning("Error generating reformulations with %s: %s", model_name, str(e))
                # Continue to fallback
                continue
                
        logger.error("All fallback models failed. Returning original query.")
        return [query]
