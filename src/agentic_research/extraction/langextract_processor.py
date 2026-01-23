"""
LangExtract processor for structured information extraction from research papers.

This module provides the LangExtractProcessor class that uses LangExtract
(or compatible LLM backends) to extract structured entities with source grounding.
"""

import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog

from agentic_research.core.config import get_settings
from agentic_research.extraction.schemas import (
    ExtractionType,
    SourceSpan,
    Extraction,
    PaperExtractions,
    PAPER_EXTRACTION_SCHEMA,
)
from agentic_research.extraction.examples import (
    get_extraction_examples,
    format_examples_for_prompt,
    EXTRACTION_PROMPT_TEMPLATE,
)

logger = structlog.get_logger(__name__)


class LangExtractProcessor:
    """
    Processor for extracting structured entities from research papers.

    Uses LangExtract library when available, with fallback to direct LLM calls
    for providers that support function calling or JSON mode.

    Supports multiple LLM providers:
    - Gemini (Google AI)
    - OpenAI (GPT-4)
    - Ollama (local models)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the LangExtract processor.

        Args:
            provider: LLM provider (gemini, openai, ollama). Uses config default if None.
            model: Model name. Uses provider-specific default if None.
        """
        self.settings = get_settings()
        self.provider = provider or self.settings.LANGEXTRACT_PROVIDER
        self.model = model or self._get_default_model()
        self.examples = get_extraction_examples()

        # Provider-specific clients (lazy initialized)
        self._gemini_client = None
        self._openai_client = None
        self._ollama_client = None

        logger.info(
            "LangExtract processor initialized",
            provider=self.provider,
            model=self.model,
        )

    def _get_default_model(self) -> str:
        """Get default model for the current provider."""
        if self.provider == "gemini":
            return self.settings.LANGEXTRACT_MODEL
        elif self.provider == "openai":
            return "gpt-4o-mini"
        elif self.provider == "ollama":
            return self.settings.OLLAMA_MODEL
        return self.settings.LANGEXTRACT_MODEL

    @property
    def gemini_client(self):
        """Lazy initialization of Gemini client."""
        if self._gemini_client is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.settings.GEMINI_API_KEY)
                self._gemini_client = genai.GenerativeModel(self.model)
                logger.debug("Gemini client initialized", model=self.model)
            except ImportError:
                logger.warning("google-generativeai not installed")
                raise ImportError("google-generativeai package required for Gemini provider")
        return self._gemini_client

    @property
    def openai_client(self):
        """Lazy initialization of OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import OpenAI

                self._openai_client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
                logger.debug("OpenAI client initialized", model=self.model)
            except ImportError:
                logger.warning("openai not installed")
                raise ImportError("openai package required for OpenAI provider")
        return self._openai_client

    @property
    def ollama_client(self):
        """Lazy initialization of Ollama client."""
        if self._ollama_client is None:
            try:
                import ollama

                self._ollama_client = ollama.Client(host=self.settings.OLLAMA_BASE_URL)
                logger.debug("Ollama client initialized", base_url=self.settings.OLLAMA_BASE_URL)
            except ImportError:
                logger.warning("ollama not installed")
                raise ImportError("ollama package required for Ollama provider")
        return self._ollama_client

    def extract_from_paper(
        self,
        text: str,
        paper_id: str,
        title: str = "",
        max_retries: int = None,
    ) -> PaperExtractions:
        """
        Extract structured entities from a research paper.

        Args:
            text: Full text of the paper
            paper_id: Unique identifier for the paper
            title: Paper title (optional)
            max_retries: Maximum retry attempts (uses config default if None)

        Returns:
            PaperExtractions object containing all extracted entities
        """
        max_retries = max_retries or self.settings.LANGEXTRACT_MAX_RETRIES

        logger.info(
            "Starting paper extraction",
            paper_id=paper_id,
            text_length=len(text),
            provider=self.provider,
        )

        # Truncate very long papers to avoid token limits
        max_chars = 50000  # ~12.5k tokens for typical text
        if len(text) > max_chars:
            logger.warning(
                "Truncating long paper",
                paper_id=paper_id,
                original_length=len(text),
                truncated_to=max_chars,
            )
            text = text[:max_chars]

        # Build extraction prompt
        examples_str = format_examples_for_prompt(self.examples, max_examples=2)
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            examples=examples_str,
            text=text,
        )

        # Attempt extraction with retries
        last_error = None
        for attempt in range(max_retries):
            try:
                raw_response = self._call_llm(prompt)
                extractions = self._parse_response(raw_response, paper_id, text)

                logger.info(
                    "Extraction successful",
                    paper_id=paper_id,
                    extraction_count=len(extractions),
                    attempt=attempt + 1,
                )

                return PaperExtractions(
                    paper_id=paper_id,
                    title=title,
                    extractions=extractions,
                    extraction_metadata={
                        "provider": self.provider,
                        "model": self.model,
                        "text_length": len(text),
                        "extracted_at": datetime.now().isoformat(),
                    },
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    "Extraction attempt failed",
                    paper_id=paper_id,
                    attempt=attempt + 1,
                    error=str(e),
                )

        # All retries failed
        logger.error(
            "All extraction attempts failed",
            paper_id=paper_id,
            max_retries=max_retries,
            last_error=str(last_error),
        )

        # Return empty extractions rather than raising
        return PaperExtractions(
            paper_id=paper_id,
            title=title,
            extractions=[],
            extraction_metadata={
                "provider": self.provider,
                "model": self.model,
                "error": str(last_error),
                "failed_at": datetime.now().isoformat(),
            },
        )

    def _call_llm(self, prompt: str) -> str:
        """
        Call the configured LLM provider.

        Args:
            prompt: The extraction prompt

        Returns:
            Raw response text from the LLM
        """
        if self.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "ollama":
            return self._call_ollama(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API for extraction."""
        response = self.gemini_client.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        )
        return response.text

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API for extraction."""
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at extracting structured information from academic papers. Always respond with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return response.choices[0].message.content

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for extraction."""
        response = self.ollama_client.generate(
            model=self.model,
            prompt=prompt,
            format="json",
            options={
                "temperature": 0.1,
                "num_ctx": 8192,
            },
        )
        return response["response"]

    def _parse_response(
        self,
        response: str,
        paper_id: str,
        original_text: str,
    ) -> List[Extraction]:
        """
        Parse LLM response into Extraction objects.

        Args:
            response: Raw LLM response (should be JSON)
            paper_id: Paper identifier
            original_text: Original paper text for source validation

        Returns:
            List of Extraction objects
        """
        # Clean response - remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response", error=str(e))
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse JSON from response: {response[:200]}")

        # Handle both {"extractions": [...]} and direct [...] formats
        raw_extractions = data.get("extractions", data) if isinstance(data, dict) else data
        if not isinstance(raw_extractions, list):
            raw_extractions = [raw_extractions]

        extractions = []
        for raw_ext in raw_extractions:
            try:
                extraction = self._parse_single_extraction(raw_ext, paper_id, original_text)
                if extraction:
                    extractions.append(extraction)
            except Exception as e:
                logger.warning(
                    "Failed to parse extraction",
                    error=str(e),
                    raw_extraction=str(raw_ext)[:200],
                )

        return extractions

    def _parse_single_extraction(
        self,
        raw: Dict[str, Any],
        paper_id: str,
        original_text: str,
    ) -> Optional[Extraction]:
        """
        Parse a single extraction from raw dictionary.

        Args:
            raw: Raw extraction dictionary from LLM
            paper_id: Paper identifier
            original_text: Original text for source validation

        Returns:
            Extraction object or None if parsing fails
        """
        # Get extraction type
        ext_type_str = raw.get("type", "").lower()
        try:
            ext_type = ExtractionType(ext_type_str)
        except ValueError:
            logger.warning(f"Unknown extraction type: {ext_type_str}")
            return None

        # Get name and content
        name = raw.get("name", "")
        content = raw.get("content", "")
        if not name or not content:
            return None

        # Parse source span
        source_span_data = raw.get("source_span", {})
        if isinstance(source_span_data, dict):
            span_text = source_span_data.get("text", content[:100])
            span_start = source_span_data.get("start", 0)
            span_end = source_span_data.get("end", len(span_text))

            # Validate span against original text if possible
            if span_text and span_text in original_text:
                actual_start = original_text.find(span_text)
                if actual_start >= 0:
                    span_start = actual_start
                    span_end = actual_start + len(span_text)
        else:
            span_text = content[:100]
            span_start = 0
            span_end = len(span_text)

        source_span = SourceSpan(
            start=span_start,
            end=span_end,
            text=span_text[:500],  # Limit text length
        )

        # Get confidence
        confidence = raw.get("confidence", 0.8)
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = 0.8
        confidence = max(0.0, min(1.0, confidence))

        # Get attributes
        attributes = raw.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}

        return Extraction(
            extraction_type=ext_type,
            name=name,
            content=content,
            source_span=source_span,
            confidence=confidence,
            attributes=attributes,
            paper_id=paper_id,
            extracted_at=datetime.now(),
        )

    def extract_methods(self, text: str, paper_id: str) -> List[Extraction]:
        """Extract only methods/techniques from paper."""
        result = self.extract_from_paper(text, paper_id)
        return result.get_methods()

    def extract_datasets(self, text: str, paper_id: str) -> List[Extraction]:
        """Extract only datasets from paper."""
        result = self.extract_from_paper(text, paper_id)
        return result.get_datasets()

    def extract_findings(self, text: str, paper_id: str) -> List[Extraction]:
        """Extract only findings from paper."""
        result = self.extract_from_paper(text, paper_id)
        return result.get_findings()

    def extract_citations(self, text: str, paper_id: str) -> List[Extraction]:
        """Extract only citations from paper."""
        result = self.extract_from_paper(text, paper_id)
        return result.get_citations()
