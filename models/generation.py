"""Text generation model for RAG pipeline."""

import torch
from typing import List, Dict, Any, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    Pipeline
)
from config import config


class TextGenerationModel:
    """Wrapper for text generation model with tokenizer."""

    def __init__(self, model_name: Optional[str] = None, hf_token: Optional[str] = None):
        """Initialize text generation model.

        Args:
            model_name: Name of the Hugging Face model to use.
                       If None, uses config default.
            hf_token: Hugging Face token for model access.
        """
        self.model_name = model_name or config.model.generator_name
        self.hf_token = hf_token or getattr(config.model, 'hf_token', None)
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load the tokenizer, model, and create generation pipeline."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                use_fast=True
            )

            # Set pad token if not present
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )

            # Create generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=config.model.max_new_tokens,
                do_sample=True,
                top_k=config.model.top_k,
                top_p=config.model.top_p,
                repetition_penalty=config.model.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load generation model {self.model_name}: {e}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt for generation.
            max_new_tokens: Maximum number of new tokens to generate.
                           If None, uses config default.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized")

        # Override config defaults with provided parameters
        gen_kwargs = {}
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens

        try:
            with torch.inference_mode():
                result = self.pipeline(prompt, **gen_kwargs)
                return result[0]["generated_text"].strip()
        except Exception as e:
            raise RuntimeError(f"Text generation failed: {e}")

    def truncate_prompt(self, prompt: str, max_length: Optional[int] = None) -> str:
        """Truncate prompt to maximum length.

        Args:
            prompt: Text to truncate.
            max_length: Maximum token length. If None, uses config default.

        Returns:
            Truncated prompt.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        max_len = max_length or config.model.input_max_length
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

        if len(tokens) > max_len:
            tokens = tokens[-max_len:]

        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def get_max_input_length(self) -> int:
        """Get the maximum input length for the model."""
        return config.model.input_max_length

    def __repr__(self) -> str:
        return f"TextGenerationModel(name='{self.model_name}')"
