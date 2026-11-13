
"""
Distillation dataset utilities for Weird AI.

Chapter 8 introduces model distillation:

    teacher model -> synthetic training examples -> student model

For Weird AI, this means preparing teacher-generated parody lyrics and teacher
notes as reusable supervised training data. This module does not train a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
import copy
import json
import random
from pathlib import Path
from typing import Any, Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]:
        ...


@dataclass
class TeacherExample:
    prompt: str
    teacher_lyrics: str
    teacher_notes: str
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillationRecord:
    prompt: str
    target_text: str
    full_text: str
    input_ids: list[int]
    labels: list[int]
    prompt_length: int
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_teacher_example(example: dict, req_fields:
                             list[str] = ["prompt", "teacher_lyrics", "teacher_notes", "quality_score"],
                             score_key: str = "quality_score") -> bool:
    """
    Validate that a raw teacher example contains usable fields.

    Required fields:
        prompt, teacher_lyrics, teacher_notes, quality_score
    """
    # TODO:
    # 1. Check that all required fields are present.
    # 2. Check that prompt, teacher_lyrics, and teacher_notes are non-empty strings.
    # 3. Check that quality_score can be converted to float.
    # 4. Return True only if all checks pass.
    for req in req_fields:
        if (req not in example):
            return False
        if (not str(example[req]).strip()):
            return False

        if (req == score_key):
            try:
                float(example[req])
            except (ValueError, TypeError):
                return False
    return True

def teacher_example_from_dict(example: dict, score_key: str = "quality_score") -> TeacherExample:
    """Convert a valid dictionary into a TeacherExample."""
    # TODO:
    # 1. Validate the example.
    # 2. Raise ValueError if invalid.
    # 3. Convert quality_score to float.
    # 4. Put any extra keys into metadata.
    # 5. Return a TeacherExample.
    req_fields = [var.name for var in fields(TeacherExample) if var.name.lower() != "metadata"]
    if (not validate_teacher_example(example, req_fields=req_fields,
                                     score_key=score_key)):
        raise ValueError(f"ValueError: Teacher example is invalid, \nExample: {example} \nRequired_fields: {req_fields}")

    example[score_key] = float(example[score_key])

    return TeacherExample(
        *[example.pop(req) for req in req_fields],
        metadata=example  # rest of example entries
    )




def format_teacher_response(
    teacher_lyrics: str,
    teacher_notes: str,
    *,
    include_thinking: bool = True,
) -> str:
    """Format teacher notes and lyrics as a distillation target."""
    # TODO:
    # 1. Strip teacher_lyrics and teacher_notes.
    # 2. If include_thinking=True, return notes inside <think>...</think> tags,
    #    followed by a blank line and the lyrics.
    # 3. If include_thinking=False, return only the lyrics.
    # 4. Avoid extra leading/trailing whitespace.
    teacher_lyrics = teacher_lyrics.strip()
    teacher_notes = teacher_notes.strip()

    return (
        (f"<think>{teacher_notes}</think>\n\n" if include_thinking else "") +
        teacher_lyrics
    )


def format_training_prompt(prompt: str) -> str:
    """Format a Weird AI prompt for distillation training."""
    # TODO:
    # 1. Strip the prompt.
    # 2. Return a consistent format such as:
    #    "User: Write a parody...\\nAssistant:"
    prompt = prompt.strip()
    return f"User: {prompt}\n\nAssistant:"


def build_full_text(prompt: str, target_text: str) -> str:
    """Combine formatted prompt and target text."""
    # TODO:
    # 1. Use format_training_prompt(prompt).
    # 2. Append target_text after the assistant marker.
    # 3. Return the full training string.
    prompt = format_training_prompt(prompt)
    return f"{prompt} {target_text}"


def tokenize_text(tokenizer: TokenizerLike, text: str) -> list[int]:
    """Tokenize text using an object with encode(text) -> list[int]."""
    # TODO:
    # 1. Call tokenizer.encode(text).
    # 2. Validate that the result is a list of integers.
    # 3. Return the token IDs.

    tokens = tokenizer.encode(text)
    if (not all(isinstance(item, int) for item in tokens)):
        raise ValueError("ValueError: Tokenizer failed to tokenize all text to integers")
    return tokens


def create_labels(
    input_ids: list[int],
    prompt_length: int,
    *,
    mask_prompt: bool = True,
    ignore_index: int = -100,
) -> list[int]:
    """
    Create training labels.

    If mask_prompt=True, prompt token labels should be replaced with ignore_index.
    """
    # TODO:
    # 1. Copy input_ids into labels.
    # 2. If mask_prompt=True, replace labels before prompt_length with ignore_index.
    # 3. Return labels.
    labels = copy.copy(input_ids)
    if (mask_prompt):
        labels[:prompt_length] = [ignore_index] * prompt_length
    return labels


def build_distillation_record(
    example: TeacherExample,
    tokenizer: TokenizerLike,
    *,
    include_thinking: bool = True,
    mask_prompt: bool = True,
) -> DistillationRecord:
    """Build one processed distillation record."""
    # TODO:
    # 1. Format the teacher response.
    # 2. Format the prompt.
    # 3. Build the full text.
    # 4. Tokenize the prompt and full text.
    # 5. Create labels.
    # 6. Return a DistillationRecord.
    prompt = format_training_prompt(example.prompt)
    target_text = format_teacher_response(example.teacher_lyrics,
                                               example.teacher_notes,
                                               include_thinking=include_thinking)
    full_text = build_full_text(prompt, target_text)

    input_ids = tokenize_text(tokenizer, full_text)
    labels = create_labels(input_ids, len(prompt), mask_prompt=mask_prompt)

    return DistillationRecord(
        prompt=example.prompt,
        full_text=full_text,
        target_text=target_text,
        input_ids=input_ids,
        labels=labels,
        prompt_length=len(prompt)
    )


def filter_teacher_examples(
    examples: list[TeacherExample],
    *,
    min_quality_score: float = 0.75,
    min_lyric_lines: int = 4
) -> list[TeacherExample]:
    """Keep only high-quality teacher examples."""
    # TODO:
    # 1. Keep examples with quality_score >= min_quality_score.
    # 2. Keep examples with at least min_lyric_lines non-empty lyric lines.
    # 3. Return the filtered list.
    filtered = []
    for example in examples:
        line_count = 0
        for line in example.teacher_lyrics.split("\\n"):
            if (line.strip()):
                line_count += 1

        if (example.quality_score >= min_quality_score and line_count >= min_lyric_lines):
            filtered.append(example)

    return filtered


def split_records(
    records: list[DistillationRecord],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[DistillationRecord], list[DistillationRecord]]:
    """Split records into train and validation sets."""
    # TODO:
    # 1. Validate validation_fraction is between 0 and 1.
    # 2. Shuffle a copy of records using seed.
    # 3. Split into train and validation records.
    # 4. Return both lists.
    if (validation_fraction <= 0 or validation_fraction >= 1):
        raise ValueError(f"ValueError: {type(validation_fraction).__name__} is invalid, {validation_fraction}.")
    random.seed(seed)
    shuffled_records = records.copy()
    random.shuffle(shuffled_records)
    validation_len = int(validation_fraction * len(shuffled_records))
    return (shuffled_records[validation_len:], shuffled_records[:validation_len])

def record_to_json_dict(record: DistillationRecord) -> dict:
    """Convert a DistillationRecord into a JSON-serializable dictionary."""
    # TODO:
    # 1. Convert the dataclass to a dictionary.
    # 2. Return the dictionary.
    return asdict(record)


def export_jsonl(records: list[DistillationRecord], path: str | Path) -> None:
    """Export records to a JSONL file."""
    # TODO:
    # 1. Open path for writing with UTF-8 encoding.
    # 2. Write one JSON object per line.
    # 3. Use record_to_json_dict for each record.
    with open(path, "w", encoding="utf-8") as file:
        json.dump([record_to_json_dict(record) for record in records], file)

def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into dictionaries."""
    # TODO:
    # 1. Open path with UTF-8 encoding.
    # 2. Read each non-empty line.
    # 3. Parse JSON and append to a list.
    # 4. Return the list.
    with open(path, "r", encoding="utf-8") as file:
        records = json.load(file)
    return records


def summarize_distillation_records(records: list[DistillationRecord]) -> str:
    """Create a readable summary of distillation records."""
    # TODO:
    # 1. Handle empty records.
    # 2. Include number of records.
    # 3. Include average input length.
    # 4. Include average prompt length.
    # 5. Include average target length.
    if (len(records) <= 0 ):
        return "No records."

    total_input_len = 0
    total_prompt_len = 0
    total_target_len = 0
    for record in records:
        total_input_len += len(record.input_ids)
        total_prompt_len += len(record.prompt)
        total_target_len += len(record.target_text)


    return f"""
        {"="*10} Summary {"="*10}
        Total records: {len(records)}
        Average input length: {total_input_len / len(records): .2f}
        Average prompt length: {total_prompt_len / len(records): .2f}
        Average target length: {total_target_len / len(records): .2f}
    """
