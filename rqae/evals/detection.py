# Taken from https://arxiv.org/abs/2410.13928 (Appendix A.3) and https://github.com/EleutherAI/sae-auto-interp/blob/0821c25bdcd4891ac5d45fde8a968ec536f801a8/sae_auto_interp/scorers/classifier/prompts/detection_prompt.py

from rqae.feature import Feature
import anthropic
import os
import random
import numpy as np
import ast
from rqae.evals.utils import display_messages

MODEL = "claude-3-5-sonnet-20241022"
# MODEL = "claude-3-5-haiku-20241022"

SYSTEM = """You are an intelligent and meticulous linguistics researcher.

You will be given a certain feature of text, such as "male pronouns" or "text with negative sentiment".

You will then be given several text examples. Your task is to determine which examples possess the feature.

For each example in turn, return 1 if the sentence is correctly labeled or 0 if the tokens are mislabeled. You must return your response in a valid Python list. Do not return anything else besides a Python list.
"""

EXAMPLE_INPUTS = [
    """Feature explanation: Words related to American football positions, specifically the tight end position.

Text examples:

Example 0:<|endoftext|>Getty ImagesĊĊPatriots tight end Rob Gronkowski had his bossâĢĻ
Example 1: names of months used in The Lord of the Rings:ĊĊâĢľâĢ¦the
Example 2: Media Day 2015ĊĊLSU defensive end Isaiah Washington (94) speaks to the
Example 3: shown, is generally not eligible for ads. For example, videos about recent tragedies,
Example 4: line, with the left side âĢĶ namely tackle Byron Bell at tackle and guard Amini
""",
    """Feature explanation: The word "guys" in the phrase "you guys".

Text examples:

Example 0: enact an individual health insurance mandate?âĢĿ, Pelosi's response was to dismiss both
Example 1: birth control access<|endoftext|> but I assure you women in Kentucky aren't laughing as they struggle
Example 2: du Soleil Fall Protection Program with construction requirements that do not apply to theater settings because
Example 3:Ċ<|endoftext|> distasteful. Amidst the slime lurk bits of Schadenfre
Example 4: the<|endoftext|>ľI want to remind you all that 10 days ago (director Massimil
""",
    """Feature explanation: "of" before words that start with a capital letter.

Text examples:

Example 0: climate, TomblinâĢĻs Chief of Staff Charlie Lorensen said.Ċ
Example 1: no wonderworking relics, no true Body and Blood of Christ, no true Baptism
Example 2:ĊĊDeborah Sathe, Head of Talent Development and Production at Film London,
Example 3:ĊĊIt has been devised by Director of Public Prosecutions (DPP)
Example 4: and fair investigation not even include the Director of Athletics? Â· Finally, we believe the
""",
]

EXAMPLE_OUTPUTS = ["[1,0,0,0,1]", "[0,0,0,0,0]", "[1,1,1,1,1]"]


def detect(
    feature: Feature,
    top_n: int = 5,
    token_radius: int = 8,
    verbose: bool = False,
):
    client = anthropic.Anthropic()
    system_prompt = SYSTEM

    few_shot = []

    random_order = random.sample(range(len(EXAMPLE_INPUTS)), len(EXAMPLE_INPUTS))
    for i in random_order:
        few_shot.append([EXAMPLE_INPUTS[i], EXAMPLE_OUTPUTS[i]])

    # sort activations
    stacked_activations = np.stack([x["activations"] for x in feature.activations])
    activations_order = stacked_activations.max(1).argsort()
    activations_order = activations_order[::-1]
    feature.activations = [feature.activations[i] for i in activations_order]

    # choose top_n activations, and top_n "bottom" (i.e. random) activations
    activation_indices = random.sample(list(range(top_n * 4)), top_n) + random.sample(
        list(range(len(feature.activations) - top_n * 4, len(feature.activations))),
        top_n,
    )
    # Shuffle
    random.shuffle(activation_indices)
    activation_indices = activation_indices[:top_n]

    user_prompt = f"Feature explanation: {feature.explanation}\n\nText examples:\n\n"
    expected_output = []
    for example_idx, activation_idx in enumerate(activation_indices):
        activation = feature.activations[activation_idx]["activations"]
        text_sequence = feature.activations[activation_idx]["text"]

        # Find index of max activation
        max_activation_idx = activation.argmax()

        # Get start and end indices within token_radius of max activation, ensuring window size
        start_idx = max(0, max_activation_idx - token_radius)
        remaining_before = max_activation_idx - start_idx
        extra_after = token_radius + (token_radius - remaining_before)
        end_idx = min(len(text_sequence), max_activation_idx + extra_after + 1)

        # Slice sequences to only include window around max activation
        activation = activation[start_idx:end_idx]
        text_sequence = text_sequence[start_idx:end_idx]

        user_prompt += f"Example {example_idx}: {''.join(text_sequence)}\n"
        expected_output.append(1 if activation_idx < top_n * 4 else 0)
    messages = []
    for few_shot_input, few_shot_output in few_shot:
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": few_shot_input}],
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": few_shot_output}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        }
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        temperature=0.5,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
        extra_headers=({"anthropic-beta": "prompt-caching-2024-07-31"}),
    )
    content = response.content[0].text
    try:
        output = ast.literal_eval(content)
    except Exception as e:
        raise ValueError(f"Anthropic returned an invalid output: {content}") from e

    score = sum([x == y for x, y in zip(output, expected_output)]) / len(
        expected_output
    )

    messages.append(
        {"role": "assistant", "content": [{"type": "text", "text": content}]}
    )
    messages.append(
        {
            "role": "GROUND_TRUTH",
            "content": [{"type": "text", "text": str(expected_output)}],
        }
    )
    messages.append(
        {
            "role": "SCORE",
            "content": [{"type": "text", "text": str(score)}],
        }
    )

    # # Example given below
    if verbose:
        print(display_messages(system_prompt, messages))

    return score, display_messages(system_prompt, messages)
