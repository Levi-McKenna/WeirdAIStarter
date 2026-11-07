"""
Self-refinement utilities for Weird AI.

This module connects Chapter 5 of Build a Reasoning Model from Scratch to the
Weird AI project.

Chapter 4 added best-of-N generation:

    generate several candidates -> evaluate each -> select the best

Chapter 5 adds self-refinement:

    generate one candidate -> evaluate -> ask for a revision -> evaluate again
    -> keep the better version -> repeat

The goal is to improve output quality during inference without retraining the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


GenerationFunction = Callable[..., str]
EvaluationFunction = Callable[[str], dict]


@dataclass
class RefinementStep:
    """Store information about one refinement attempt."""

    iteration: int
    prompt: str
    previous_text: str
    revised_text: str
    previous_score: float
    revised_score: float
    accepted: bool
    evaluation: dict = field(default_factory=dict)


def get_score(evaluation: dict, score_key: str = "overall_score") -> float:
    """Extract a numeric score from an evaluation dictionary."""

    # TODO:
    # 1. Read score_key from the evaluation dictionary.
    # 2. Convert it to a float.
    # 3. If the key is missing or cannot be converted, return 0.0.

    if (score_key not in evaluation):
        return 0.0

    try:
        return float(evaluation[score_key])
    except ValueError:
        print(f"ValueError: {type(evaluation).__name__}[{score_key}] is not a valid value")
        return 0.0
    except TypeError:
        print(f"TypeError: {type(evaluation).__name__}[{score_key}] cannot be casted to a float")
        return 0.0


def build_refinement_prompt(
    original_prompt: str,
    current_text: str,
    evaluation: dict,
    *,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    score_key: str = "overall_score",
) -> str:
    """Build a prompt asking the model to improve a generated parody."""

    # TODO:
    # 1. Include the original prompt or topic.
    # 2. Include the current lyrics.
    # 3. Include the current overall score.
    # 4. Include any available evaluation details, such as rhyme, syllables, and structure.
    # 5. Include focus items, such as stronger rhymes or more consistent syllables.
    # 6. Include preserve items, such as topic, tone, and approximate length.
    # 7. Ask the model to return only the revised lyrics.
    #
    # Keep the prompt specific. "Make it better" is too vague.

    return f"""
        Problem: {original_prompt}.

        Current lyrics: {current_text}.
        Current overall score: {get_score(evaluation, score_key=score_key)},
        Rhyme score: {evaluation["rhyme_score"]}
        Syllable consistency score: {evaluation["syllable_consistency_score"]}
        Structure score: {evaluation["structure_score"]}

        Focus items:
            {"\n".join([f"- {req}" for req in focus]) if focus else ""}

        Preserve items:
            {"\n".join([f"- {req}" for req in preserve]) if preserve else ""}

         {(
        "Revise and improve the lyrics in order to increase the scores,"
        "particularly focus on the Focus items while preserving the Preserve"
        "items. Produce only the revised lyrics."
        )}
        Revised lyrics:
    """


def refine_once(
    original_prompt: str,
    current_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> RefinementStep:
    """Run one self-refinement step."""

    # TODO:
    # 1. Evaluate current_text.
    # 2. Build a refinement prompt using build_refinement_prompt.
    # 3. Generate revised text with generation_func.
    # 4. Evaluate the revised text.
    # 5. Compare the old score and revised score.
    # 6. Return a RefinementStep.
    #
    # This function should not decide whether to continue the full loop.
    # It only performs one attempt.

    evaluation = evaluation_func(current_text)
    refinement_prompt = build_refinement_prompt(original_prompt, current_text,
                                                evaluation, focus=focus,
                                                preserve=preserve, score_key=score_key)
    revised_text = generation_func(refinement_prompt, temperature=temperature,
                                   top_p=top_p, max_new_tokens=max_new_tokens)
    revise_eval = evaluation_func(revised_text)

    return RefinementStep (
        1,
        original_prompt,
        current_text,
        revised_text,
        get_score(evaluation, score_key=score_key),
        get_score(revise_eval, score_key=score_key),
        True,
        revise_eval
    )


def choose_better_text(
    current_text: str,
    revised_text: str,
    current_evaluation: dict,
    revised_evaluation: dict,
    *,
    score_key: str = "overall_score",
    require_improvement: bool = True,
) -> tuple[str, dict, bool]:
    """Decide whether to accept a revised text."""

    # TODO:
    # 1. Compare the current score and revised score.
    # 2. If require_improvement is True, accept only strictly higher scores.
    # 3. If require_improvement is False, accept equal scores as well.
    # 4. Return the selected text, selected evaluation, and whether the revision was accepted.

    accepted = (
        get_score(revised_evaluation, score_key=score_key) > get_score(current_evaluation, score_key=score_key)
        if require_improvement 
        else get_score(revised_evaluation, score_key=score_key) >= get_score(current_evaluation, score_key=score_key)
    )

    if (not accepted):
        return current_text, current_evaluation, accepted
    return revised_text, revised_evaluation, accepted


def should_continue(
    *,
    iteration: int,
    max_iterations: int,
    current_score: float,
    target_score: Optional[float] = None,
    last_improved: bool = True,
    stop_on_no_improvement: bool = True,
) -> bool:
    """Decide whether the refinement loop should continue."""

    # TODO:
    # 1. Stop if iteration has reached max_iterations.
    # 2. Stop if target_score is set and current_score is high enough.
    # 3. Stop if stop_on_no_improvement is True and last_improved is False.
    # 4. Otherwise continue.
    if (iteration >= max_iterations):
        return False
    if (target_score and current_score >= target_score):
        return False
    if (stop_on_no_improvement and not last_improved):
        return False
    return True


def iterative_refinement(
    original_prompt: str,
    initial_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    max_iterations: int = 3,
    target_score: Optional[float] = None,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    stop_on_no_improvement: bool = True,
    require_improvement: bool = True,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> dict:
    """Run the full self-refinement loop."""

    # TODO:
    # 1. Evaluate initial_text.
    # 2. Repeatedly call refine_once until should_continue says to stop.
    # 3. After each step, keep the better text using choose_better_text.
    # 4. Track every RefinementStep in a list.
    # 5. Return a result dictionary with the initial and final information.
    #
    # Be careful to avoid infinite loops.

    if (max_iterations <= 0):
        raise ValueError(f"{type(max_new_tokens).__name__} must be greater than 0")

    init_eval = evaluation_func(initial_text)
    current_eval = init_eval
    current_text = initial_text
    iteration = 1
    last_improved = True
    steps = []
    while (should_continue(iteration=iteration, max_iterations=max_iterations,
                           current_score=get_score(current_eval, score_key=score_key),
                           target_score=target_score,
                           last_improved=last_improved,
                           stop_on_no_improvement=stop_on_no_improvement)):
        refine_step = refine_once(original_prompt, initial_text,
                                  generation_func, evaluation_func,
                                  temperature=temperature, top_p=top_p,
                                  max_new_tokens=max_new_tokens,
                                  score_key=score_key, focus=focus,
                                  preserve=preserve)
        steps.append(refine_step)
        best_text, best_eval, last_improved = choose_better_text(current_text, refine_step.revised_text,
                                                                   current_eval, refine_step.evaluation,
                                                                   score_key=score_key, require_improvement=require_improvement)
        refine_step.iteration = iteration
        # hacky but fixes the fact that answers arent compared properly in
        # refine_once
        refine_step.accepted = last_improved

        current_text, current_eval = best_text, best_eval
        iteration += 1


    stopped_for = []
    if (iteration >= max_iterations):
        stopped_for.append(f"max_iterations")
    if (target_score and get_score(current_eval, score_key=score_key) >= target_score):
        stopped_for.append(
            f"target_score"
        )
    if (stop_on_no_improvement and not last_improved):
        stopped_for.append(
            f"no_improvement"
        )

    return {
        "initial_text": initial_text,
        "initial_score": get_score(init_eval, score_key=score_key),
        "initial_evaluation": init_eval,
        "final_text": current_text,
        "final_score": get_score(current_eval, score_key=score_key),
        "final_evaluation": current_eval,
        "iterations": steps,
        "stopped_reason": stopped_for[0],
        "reasons_stopped": stopped_for
    }


def summarize_refinement(result: dict) -> str:
    """Create a readable summary of a refinement run."""

    # TODO:
    # 1. Include the initial score and final score.
    # 2. Include the number of iterations.
    # 3. Include whether each iteration was accepted or rejected.
    # 4. Include the stopped reason.
    #
    # This is useful for debugging and for showing the tradeoff between extra
    # inference compute and output improvement.

    return f"""
        {"="*10} Refinement Process {"="*10}

        Initial score: {result["initial_score"]}
        Final score: {result["final_score"]}
        
        Iterations: {len(result["iterations"])}
            {
            "\n".join(
                [f"\t- Step {idx}: {"Accepted" if step.accepted else "Rejected"}"
                    for idx, step in enumerate(result["iterations"], start=1)]
            )}
        
        Stopped reasons:
            {
            "\n".join(
                [f"\t- {idx}. {reason}"
                 for idx, reason in enumerate(result["reasons_stopped"], start=1)] 
            ) if "reasons_stopped" in result else result["stopped_reason"]
            }

        {"="*10} {"="*10}
    """
