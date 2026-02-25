"""
Prompting utilities for multiple-choice QA.
Example submission.
"""

import torch
from torch import Tensor
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import softmax


class PromptTemplate:
    TEMPLATES = {
        "basic": "Context: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_formatted}\n\nAnswer:",
        "instruction": "Read the following passage and answer the question.\n\nPassage: {context}\n\nQuestion: {question}\n\n{choices_formatted}\n\nSelect the letter:",
        "simple": "{context}\n{question}\n{choices_formatted}\nThe answer is",
    }

    def __init__(
        self,
        template_name: str = "basic",
        custom_template: Optional[str] = None,
        choice_format: str = "letter",
    ):
        self.template = (
            custom_template
            if custom_template
            else self.TEMPLATES.get(template_name, self.TEMPLATES["basic"])
        )
        self.choice_format = choice_format

    def _format_choices(self, choices: List[str]) -> str:
        labels = (
            ["A", "B", "C", "D", "E", "F", "G", "H"]
            if self.choice_format == "letter"
            else [str(i + 1) for i in range(len(choices))]
        )
        return "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))

    def format(self, context: str, question: str, choices: List[str], **kwargs) -> str:
        return self.template.format(
            context=context,
            question=question,
            choices_formatted=self._format_choices(choices),
            **kwargs,
        )

    def format_with_answer(
        self, context: str, question: str, choices: List[str], answer_idx: int
    ) -> str:
        prompt = self.format(context, question, choices)
        label = (
            chr(ord("A") + answer_idx)
            if self.choice_format == "letter"
            else str(answer_idx + 1)
        )
        return f"{prompt} {label}"


class PromptingPipeline:
    def __init__(
        self,
        model,
        tokenizer,
        template: Optional[PromptTemplate] = None,
        device: str = "cuda",
    ):
        self.model = model.to(device) if hasattr(model, "to") else model
        self.tokenizer = tokenizer
        self.template = template or PromptTemplate("basic")
        self.device = device
        self.max_context_length = getattr(self.model, "context_length", None)
        self._setup_choice_tokens()

    def _setup_choice_tokens(self):
        self.choice_tokens = {}
        self.choice_token_sequences = {}
        for label in ["A", "B", "C", "D"]:
            # Prefer space-prefixed forms since prompts typically end with "... answer is".
            for candidate in [f" {label}", label]:
                token_ids = self.tokenizer.encode(candidate)
                if token_ids:
                    self.choice_tokens[label] = token_ids[0]
                    self.choice_token_sequences[label] = token_ids
                    break

    @torch.no_grad()
    def predict_single(
        self,
        context: str,
        question: str,
        choices: List[str],
        return_probs: bool = False,
    ):
        self.model.eval()
        prompt = self.template.format(context, question, choices)
        prompt_ids = self.tokenizer.encode(prompt)

        # Keep the most recent tokens to match the model context window.
        if (
            self.max_context_length is not None
            and len(prompt_ids) > self.max_context_length
        ):
            prompt_ids = prompt_ids[-self.max_context_length :]

        choice_labels = ["A", "B", "C", "D"][: len(choices)]
        choice_scores = []

        for label in choice_labels:
            label_tokens = self.choice_token_sequences.get(label)
            if not label_tokens:
                choice_scores.append(float("-inf"))
                continue

            running_ids = prompt_ids.copy()
            score = 0.0
            valid = True

            # Score each answer label as an autoregressive token sequence.
            for token_id in label_tokens:
                if (
                    self.max_context_length is not None
                    and len(running_ids) > self.max_context_length
                ):
                    running_ids = running_ids[-self.max_context_length :]

                input_ids = torch.tensor([running_ids], device=self.device)
                logits = self.model(input_ids)[:, -1, :]
                vocab_size = logits.size(-1)
                if token_id < 0 or token_id >= vocab_size:
                    valid = False
                    break

                score += torch.log_softmax(logits, dim=-1)[0, token_id].item()
                running_ids.append(token_id)

            choice_scores.append(score if valid else float("-inf"))

        choice_scores = torch.tensor(choice_scores)
        if torch.isfinite(choice_scores).any():
            probs = softmax(choice_scores, dim=-1)
        else:
            probs = torch.full_like(choice_scores, 1.0 / len(choice_scores))
        prediction = probs.argmax().item()

        if return_probs:
            return prediction, probs.tolist()
        return prediction

    @torch.no_grad()
    def predict_batch(
        self, examples: List[Dict[str, Any]], batch_size: int = 8
    ) -> List[int]:
        return [
            self.predict_single(ex["context"], ex["question"], ex["choices"])
            for ex in examples
        ]


def evaluate_prompting(
    pipeline, examples: List[Dict[str, Any]], batch_size: int = 8
) -> Dict[str, Any]:
    predictions = pipeline.predict_batch(examples, batch_size)
    correct = sum(
        1
        for p, ex in zip(predictions, examples)
        if ex.get("answer", -1) >= 0 and p == ex["answer"]
    )
    total = sum(1 for ex in examples if ex.get("answer", -1) >= 0)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "predictions": predictions,
    }
