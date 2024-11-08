# Taken from https://arxiv.org/abs/2410.13928 (Appendix A.3) and https://github.com/EleutherAI/sae-auto-interp/blob/main/sae_auto_interp/scorers/classifier/prompts/fuzz_prompt.py

from rqae.feature import Feature
import anthropic
import os
import random
import numpy as np
import ast
from rqae.evals.utils import display_messages

MODEL = "claude-3-5-sonnet-20241022"
# MODEL = "claude-3-5-sonnet-20240620"
# MODEL = "claude-3-5-haiku-20241022"

SYSTEM = """You are an intelligent and meticulous linguistics researcher.

You will be given a certain feature of text, such as "male pronouns" or "text with negative sentiment". You will be given a few examples of text that contain this feature. Portions of the sentence which strongly represent this feature are between tokens << and >>.

Some examples might be mislabeled. Your task is to determine if every single token within << and >> is correctly labeled. Consider that all provided examples could be correct, none of the examples could be correct, or a mix. An example is only correct if every marked token is representative of the feature

For each example in turn, return 1 if the sentence is correctly labeled or 0 if the tokens are mislabeled. You must return your response in a valid Python list. Do not return anything else besides a Python list.
"""

EXAMPLE_INPUTS = [
    """Feature explanation: Words related to American football positions, specifically the tight end position.

Text examples:

Example 0:<|endoftext|>Getty ImagesĊĊPatriots<< tight end>> Rob Gronkowski had his bossâĢĻ
Example 1: posted<|endoftext|>You should know this<< about>> offensive line coaches: they are large, demanding<< men>>
Example 2: Media Day 2015ĊĊLSU<< defensive>> end Isaiah Washington (94) speaks<< to the>>
Example 3:<< running backs>>," he said. .. Defensive<< end>> Carroll Phillips is improving and his injury is
Example 4:<< line>>, with the left side âĢĶ namely<< tackle>> Byron Bell at<< tackle>> and<< guard>> Amini
""",
    """Feature explanation: The word "guys" in the phrase "you guys".

Text examples:

Example 0: if you are<< comfortable>> with it. You<< guys>> support me in many other ways already and
Example 1: birth control access<|endoftext|> but I assure you<< women>> in Kentucky aren't laughing as they struggle
Example 2:âĢĻs gig! I hope you guys<< LOVE>> her, and<< please>> be nice,
Example 3:American, told<< Hannity>> that âĢľyou<< guys>> are playing the race card.âĢĿ
Example 4:<< the>><|endoftext|>ľI want to<< remind>> you all that 10 days ago (director Massimil
""",
    """Feature explanation: "of" before words that start with a capital letter.

Text examples:

Example 0: climate, TomblinâĢĻs Chief<< of>> Staff Charlie Lorensen said.Ċ
Example 1: no wonderworking relics, no true Body and Blood<< of>> Christ, no true Baptism
Example 2:ĊĊDeborah Sathe, Head<< of>> Talent Development and Production at Film London,
Example 3:ĊĊIt has been devised by Director<< of>> Public Prosecutions (DPP)
Example 4: and fair investigation not even include the Director<< of>> Athletics? Â· Finally, we believe the
""",
]

EXAMPLE_OUTPUTS = ["[1,0,0,1,1]", "[0,0,0,0,0]", "[1,1,1,1,1]"]


def fuzz(
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

    activation_indices = list(range(top_n * 2))
    # Shuffle
    random.shuffle(activation_indices)
    activation_indices = activation_indices[:top_n]
    answer_key = [1] * top_n + [0] * top_n
    random.shuffle(answer_key)
    answer_key = answer_key[:top_n]

    user_prompt = f"Feature explanation: {feature.explanation}\n\nText examples:\n\n"
    expected_output = []
    for example_idx, activation_idx in enumerate(activation_indices):
        activation = feature.activations[activation_idx]["activations"]
        text_sequence = feature.activations[activation_idx]["text"]
        answer = answer_key[example_idx]

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

        nonzero_activations = [
            (text_sequence[i], int(activation[i]))
            for i in range(len(text_sequence))
            if activation[i] > 0
        ]

        if answer == 1:
            # Surround every contiguous sequence of tokens with non-zero activations with delimiters
            is_open = False
            new_text_sequence = []
            for j in range(len(text_sequence)):
                if activation[j] == activation.max() and not is_open:
                    new_text_sequence.append("<<")
                    is_open = True
                elif is_open and (
                    activation[j] != activation.max() or j == len(text_sequence) - 1
                ):
                    new_text_sequence.append(">>")
                    is_open = False
                new_text_sequence.append(text_sequence[j])
            text_sequence = new_text_sequence
            text_sequence = text_sequence[1:]  # remove <bos>
        else:
            # Surround two random contiguous sequence of tokens with zero activations with delimiters
            is_open = False
            contiguous_sequence_length = 0
            new_text_sequence = []
            zero_token_indices = activation.argsort()[:5].tolist()
            random.shuffle(zero_token_indices)
            random_zero_token_indices = zero_token_indices[:2]

            for j in range(len(text_sequence)):
                if j in random_zero_token_indices and not is_open:
                    new_text_sequence.append("<<")
                    is_open = True
                    contiguous_sequence_length = 0
                elif is_open and (
                    activation[j] > 0
                    or contiguous_sequence_length > 3
                    or j == len(text_sequence) - 1
                ):
                    new_text_sequence.append(">>")
                    is_open = False
                new_text_sequence.append(text_sequence[j])
                contiguous_sequence_length += 1
            text_sequence = new_text_sequence
            # text_sequence = text_sequence[1:]  # remove <bos>

        user_prompt += f"Example {example_idx}: {''.join(text_sequence)}\n"
        expected_output.append(answer)
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
    if "\n" in content:
        content = content.split("\n")[0]
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

    if verbose:
        print(display_messages(system_prompt, messages))

    return score, display_messages(system_prompt, messages)


"""
For convenience, here is a full output:

"""
