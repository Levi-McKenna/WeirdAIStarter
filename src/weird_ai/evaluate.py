"""
Evaluation helpers for Weird AI parody outputs.

Lesson 10 connects the verifier idea from reasoning-model evaluation to the
Weird AI project. Instead of checking whether a math answer is symbolically
correct, these functions check measurable features of generated parody lyrics:

- Can we extract the final parody from a longer model response?
- Can we normalize the lyrics into clean lines?
- Do adjacent lines rhyme?
- Are the line count and syllable counts reasonably consistent?

These scores are not a complete measure of creativity or humor. They are simple,
deterministic checks that make automated evaluation possible.
"""

from __future__ import annotations

from typing import Any

from weird_ai.rhyme_tools import analyze_lyrics


FINAL_PARODY_MARKER = "Final Parody:"


def extract_final_parody(text: str) -> str:
    """
    Extract the final parody section from a model response.

    Reasoning-style prompts often produce planning text before the final answer.
    For evaluation, we usually want to score only the final generated lyrics.

    Args:
        text: Full model response text.

    Returns:
        Text after the final "Final Parody:" marker. If the marker is missing,
        return the original text with leading/trailing whitespace removed.
    """

    # TODO:
    # 1. Handle empty or whitespace-only input safely.
    # 2. Find the last occurrence of FINAL_PARODY_MARKER in the text.
    # 3. If the marker exists, return everything after it, stripped.
    # 4. If the marker does not exist, return the original text, stripped.
    if (not text):
        return ""
    start_ans = text.rfind(FINAL_PARODY_MARKER)
    if (start_ans == -1):
        return text.strip()

    return text[start_ans+len(FINAL_PARODY_MARKER):].strip()


def normalize_lyrics(text: str) -> list[str]:
    """
    Normalize raw lyric text into clean lyric lines.

    Args:
        text: Raw lyric text.

    Returns:
        A list of non-empty lines with surrounding whitespace removed.
    """

    # TODO:
    # 1. Split the text into lines.
    # 2. Strip whitespace from each line.
    # 3. Remove blank lines.
    # 4. Return the cleaned list of lines.

    lines = text.split("\n")
    return [line.strip() for line in lines if line.strip()]


def calculate_rhyme_score(text: str) -> float:
    """
    Calculate the adjacent-line rhyme score for a parody.

    This function should delegate the rhyme work to rhyme_tools.analyze_lyrics.

    Args:
        text: Lyric text to evaluate.

    Returns:
        A float from 0.0 to 1.0.
    """

    # TODO:
    # 1. Extract the final parody from the text.
    # 2. Analyze the lyrics using analyze_lyrics().
    # 3. Return the "pairwise_rhyme_score" value as a float.
    final_parod = extract_final_parody(text)
    return analyze_lyrics(final_parod)["pairwise_rhyme_score"]


def calculate_structure_score(text: str, target_line_count: int = 4) -> float:
    """
    Score whether a parody has a reasonable structure.

    This score combines:
    - line count similarity to the target line count
    - syllable consistency across lines

    Args:
        text: Lyric text to evaluate.
        target_line_count: Desired number of non-empty lyric lines.

    Returns:
        A float from 0.0 to 1.0.
    """

    # TODO:
    # 1. Extract the final parody from the text.
    # 2. Normalize the lyrics into clean lines.
    # 3. If there are no lines, return 0.0.
    # 4. Use analyze_lyrics() to get the syllable consistency score.
    # 5. Calculate a line count score:
    #       1.0 - min(1.0, abs(actual_line_count - target_line_count) / target_line_count)
    # 6. Return the average of the line count score and syllable consistency score.
    # 7. Clamp the final result between 0.0 and 1.0.

    final_parod = extract_final_parody(text)
    norm_parod = normalize_lyrics(final_parod)
    if (len(norm_parod) < 1):
        return 0.0
    syl_score = analyze_lyrics(final_parod)["syllable_consistency_score"]
    line_score = 1.0 - min(1.0, abs(len(norm_parod) - target_line_count) / target_line_count)

    # hopefully this is correct, it really isn't though
    return min(1.0, (syl_score + line_score) / 2)


def evaluate_parody(text: str, target_line_count: int = 4) -> dict[str, Any]:
    """
    Run the full Weird AI parody evaluation pipeline.

    Args:
        text: Full model response or raw lyric text.
        target_line_count: Desired number of lyric lines.

    Returns:
        A dictionary containing extracted lyrics, analysis details, component
        scores, an overall score, and a simple pass/fail recommendation.
    """

    # TODO:
    # 1. Extract the final parody.
    # 2. Normalize the lyrics.
    # 3. Analyze the extracted parody using analyze_lyrics().
    # 4. Calculate rhyme_score using calculate_rhyme_score().
    # 5. Calculate structure_score using calculate_structure_score().
    # 6. Calculate overall_score as the average of rhyme_score and structure_score.
    # 7. Set passed to True when overall_score >= 0.60 and there is at least one lyric line.
    # 8. Return a dictionary with these keys:
    #       "extracted_text"
    #       "lines"
    #       "line_count"
    #       "rhyme_scheme"
    #       "syllable_counts"
    #       "rhyme_score"
    #       "structure_score"
    #       "overall_score"
    #       "passed"
    final_parod = extract_final_parody(text)
    norm_parod = normalize_lyrics(final_parod)
    ana_dict = analyze_lyrics(final_parod)
    rhyme_score = calculate_rhyme_score(text)
    structure_score = calculate_structure_score(text)
    overall_score = (structure_score + rhyme_score) / 2
    passed = overall_score >= 0.60 and len(norm_parod) >= 1

    return {
        "extracted_text": final_parod,
        "lines": norm_parod,
        "line_count": ana_dict["line_count"],
        "rhyme_scheme": ana_dict["rhyme_scheme"],
        "syllable_counts": ana_dict["syllable_counts"],
        "rhyme_score": rhyme_score,
        "structure_score": structure_score,
        "overall_score": overall_score,
        "passed": passed
    }
