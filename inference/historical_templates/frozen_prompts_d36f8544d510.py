from __future__ import annotations

import re
from collections import Counter

from .data import Example
from .metrics import INVALID


def classification_system(labels: list[str]) -> str:
    classes = ", ".join(labels)
    return (
        "As an expert system for classifying software requirements, your task is "
        "to carefully analyze each given requirement and assign it to exactly one "
        f"of the following classes:\n{classes}\n"
        "Output only the label of the class that corresponds to the appropriate "
        "class. Do not provide any additional text, definitions, or justification."
    )


def classification_user(definitions: str, demos: list[Example], text: str) -> str:
    rendered = "\n\n".join(
        f"Requirement: {example.text}\nLabel: {example.label}" for example in demos
    )
    return f"""Definitions:
{definitions}

Examples:
{rendered}

Requirement: {text}

Using the Definitions and Examples above, classify the requirement and provide the final label in the format:
"Label: {{class}}"."""


def classification_user_zero_shot(definitions: str, text: str) -> str:
    """Appendix-F F.2.1 template used by the primary BT-APE protocol."""
    return f"""Definitions:
{definitions}

Requirement: {text}

Using the Definitions above, classify the requirement and provide the final label in the format:
"Label: {{class}}"."""


def extract_label(response: str, labels: list[str]) -> str:
    text = (response or "").strip()
    exact = text.strip("\"' .")
    exact_matches = [label for label in labels if exact.casefold() == label.casefold()]
    if len(exact_matches) == 1:
        return exact_matches[0]

    parsed_lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*label\s*:\s*(.+?)\s*$", line, flags=re.I)
        if match:
            candidate = match.group(1).strip().strip("\"' .")
            for label in labels:
                if candidate.casefold() == label.casefold():
                    parsed_lines.append(label)
    if len(parsed_lines) == 1:
        return parsed_lines[0]
    return INVALID


def majority(votes: list[str]) -> str:
    if not votes:
        return INVALID
    label, count = Counter(votes).most_common(1)[0]
    return label if count > len(votes) // 2 else INVALID


def pe2_meta_prompt(
    current_definitions: str,
    history: list[dict[str, object]],
    errors: list[dict[str, object]],
    labels: list[str],
) -> tuple[str, str]:
    """Adapt PE2's two-step/full-template meta-prompt to Appendix-B Algorithm 2."""
    system = (
        "You are a highly capable, thoughtful, and precise prompt engineer "
        "specializing in requirements classification."
    )
    history_block = "\n\n".join(
        f"### Historical Prompt {rank}\n"
        f"Validation weighted-F1: {entry['val_weighted_f1']:.12f}\n"
        f"Iteration: {entry['iteration']}\n"
        f"Definitions:\n{entry['definitions']}"
        for rank, entry in enumerate(history, start=1)
    ) or "(no prior prompt available)"
    errors_block = "\n\n".join(
        f"### Error {index}\n"
        f"Input: {error['text']}\n"
        f"Prediction: {error['prediction']}\n"
        f"Ground-truth label: {error['label']}"
        for index, error in enumerate(errors, start=1)
    ) or "(no misclassification was available)"
    full_template = classification_user("{Definitions}", [], "{requirement text}")

    user = f"""A prompt is a set of class definitions used inside a fixed template to classify software requirements. We will refine only the Definitions; the task framing, examples, and required output format remain fixed.

The candidate labels are: {', '.join(labels)}.

## Step 1: diagnose
Study the current Definitions, their full deployment template, the top historical Definitions and validation scores, and the observed errors. Ground-truth labels are authoritative. Determine which boundary rules are missing, overlapping, misleading, or over-specific. Prefer general decision rules over copying wording from individual examples.

## Current Definitions
{current_definitions}

## Full deployment template
```
{full_template}
```

## Top historical prompts
{history_block}

## Misclassified examples
{errors_block}

For every error, reason about: (1) whether the output disagrees with the label; (2) whether the model followed the current Definitions; (3) whether the Definitions describe the input-label relationship precisely; and (4) what general, actionable edit would prevent the error without overfitting.

## Step 2: revise
Privately integrate the diagnosis into one revised set of Definitions. You may add, delete, shorten, rephrase, or reorganize definition content, but must retain exactly one definition for every candidate label. Do not add demonstrations, change the labels, change the fixed classification template, or mention validation scores and individual examples in the revised Definitions.

Return exactly one candidate wrapped in these delimiter tags and no text outside them:
<revised_prompt>
the complete revised Definitions
</revised_prompt>"""
    return system, user


def parse_revised_prompt(response: str) -> str | None:
    matches = re.findall(
        r"<revised_prompt>\s*(.*?)\s*</revised_prompt>",
        response or "",
        flags=re.I | re.S,
    )
    if len(matches) != 1:
        return None
    candidate = matches[0].strip()
    return candidate if len(candidate) >= 20 else None


def btape_meta_prompt(
    current_definitions: str,
    examples: list[dict[str, object]],
    labels: list[str],
) -> tuple[str, str]:
    """Render the BT-APE Appendix-F proposer prompt.

    Unlike the protocol-matched PE2 prompt, Algorithm 1 conditions only on the
    current definitions and the four dynamically selected examples.  It does
    not expose validation scores or a top-n history to the proposer.
    """
    correct = [row for row in examples if row["label"] == row["prediction"]]
    incorrect = [row for row in examples if row["label"] != row["prediction"]]

    def render(rows: list[dict[str, object]]) -> str:
        if not rows:
            return "(none)"
        return "\n\n".join(
            f"Requirement: {row['text']}\n"
            f"True class: {row['label']}\n"
            f"Predicted class: {row['prediction']}"
            for row in rows
        )

    system = (
        "You are a highly capable, thoughtful, and precise prompt engineer "
        "specializing in requirements classification."
    )
    user = f"""Background: A LLM has previously used the following Definitions to automatically classify requirements into the candidate classes {', '.join(labels)}. Based on these definitions, it correctly classified some requirements but misclassified other requirements.

Definitions:
{current_definitions}

Task: Your task is to enhance the Definitions of the classes {', '.join(labels)} to improve the classification accuracy of the large language model. You are given example results in the form [requirement text; true class; predicted class] to guide you in enhancing the definitions.

A set of example requirements that were correctly classified based on the definitions:
{render(correct)}

A set of example requirements that were incorrectly classified based on the definitions:
{render(incorrect)}

You may add content, shorten, or rephrase the definitions to improve clarity and accuracy.

(1) Study the Definitions: Understand the provided definitions of the classes.
(2) Analyse Examples: Analyze correctly classified and misclassified examples.
(3) Success analysis: For each correctly classified example identify why the original definitions were successful. Ask yourself: Do certain words or phrases consistently lead to correct classification? Which parts of the definitions are clear, complete, and unambiguous?
(4) Error analysis: For each misclassified example identify why the original definitions failed. Ask yourself: Do certain words or phrases consistently mislead? Are the definitions ambiguous, incomplete, or overlapping?
(5) Enhance Definitions: Enhance the definitions by adding, reducing, or rephrasing content.
(6) Enhancement Procedure: For problematic classes, add discriminative features, create exclusion clauses, use comparative framing, simplify if needed, and rephrase misleading terminology if needed.
(7) Validation Constraints: Ensure the revised definitions would prevent the same errors from recurring while maintaining generalizability through clear principles rather than example-specific fixes.

Output the enhanced Definitions only, with no additional text."""
    return system, user


def parse_btape_definitions(response: str) -> str | None:
    """Parse Appendix-F output without inventing delimiter requirements."""
    text = (response or "").strip()
    if not text:
        return None
    delimited = parse_revised_prompt(text)
    if delimited is not None:
        return delimited
    return text if len(text) >= 20 else None
