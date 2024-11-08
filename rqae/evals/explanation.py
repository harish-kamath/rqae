# Taken from https://arxiv.org/abs/2410.13928 (Appendix A.1) and https://github.com/EleutherAI/sae-auto-interp/blob/8ecfc02986f61595c53dd1befa15c0cd88cf3ec8/sae_auto_interp/explainers/default/prompts.py#L3
from rqae.feature import Feature
import anthropic
import os
import random
import numpy as np
from rqae.evals.utils import display_messages

MODEL = "claude-3-5-sonnet-20241022"
# MODEL = "claude-3-5-haiku-20241022"
# We didn't see a big difference between sonnet and haiku, so we use haiku by default for costs

SYSTEM = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire sequence of tokens will be contained between delimiters <<just like this>>. How important each token is for the behavior is listed after each example in parentheses.

- Try to produce a concise final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Do not mention the marker tokens (<< >>) in your explanation.
- Do not make lists of possible explanations. Keep your explanations short and concise.
- The last line of your response must be the formatted explanation, using [EXPLANATION]:
"""

EXAMPLE_INPUTS = [
    """
Example 1:  and he was <<over the moon>> to find
Activations: ("over", 5), (" the", 6), (" moon", 9)
Example 2:  we'll be laughing <<till the cows come home>>! Pro
Activations: ("till", 5), (" the", 5), (" cows", 8), (" come", 8), (" home", 8)
Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd
Activations: ("than", 5), (" meets", 7), (" the", 6), (" eye", 8)
""",
    """
Example 1:  a river is wide but the ocean is wid<<er>>. The ocean
Activations: ("er", 8)
Example 2:  every year you get tall<<er>>," she
Activations: ("er", 2)
Example 3:  the hole was small<<er>> but deep<<er>> than the
Activations: ("er", 9), ("er", 9)
""",
    """
Example 1:  something happening inside my <<house>>", he
Activations: ("house", 7)
Example 2:  presumably was always contained in <<a box>>", according
Activations: ("a", 5), ("box", 9)
Example 3:  people were coming into the <<smoking area>>".

However he
Activations: ("smoking", 2), ("area", 4)
Example 4:  Patrick: "why are you getting in the << way?>>" Later,
Activations: ("way", 4), ("?", 2)
""",
]

EXAMPLE_OUTPUTS = [
    "[EXPLANATION]: Common idioms in text conveying positive sentiment.",
    "[EXPLANATION]: The token 'er' at the end of a comparative adjective describing size.",
    "[EXPLANATION]: Nouns representing a distinct objects that contains something, sometimes preciding a quotation mark.",
]


def explain(
    feature: Feature,
    top_n: int = 8,  # paper mentions random/quantile sampling is better, but we could not replicate
    token_radius: int = 16,
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

    # choose top N activations
    user_prompt = ""
    for i in range(top_n):
        activation = feature.activations[i]["activations"]
        text_sequence = feature.activations[i]["text"]

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

        # Surround every contiguous sequence of tokens with non-zero activations with delimiters
        is_open = False
        new_text_sequence = []
        for j in range(len(text_sequence)):
            if activation[j] > 0 and not is_open:
                new_text_sequence.append("<<")
                is_open = True
            elif is_open and (activation[j] == 0 or j == len(text_sequence) - 1):
                new_text_sequence.append(">>")
                is_open = False
            new_text_sequence.append(text_sequence[j])
        text_sequence = new_text_sequence
        text_sequence = text_sequence[1:]  # remove <bos>

        user_prompt += f"Example {i + 1}:  {''.join(text_sequence)}\n"
        activation_str = ", ".join(
            f'("{token}", {activation})' for token, activation in nonzero_activations
        )
        user_prompt += f"Activations: {activation_str}\n"

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

    messages.append(
        {"role": "assistant", "content": [{"type": "text", "text": content}]}
    )

    # # Example given below
    if verbose:
        print(display_messages(system_prompt, messages))

    if not "[EXPLANATION]:" in content:
        raise ValueError(f"Anthropic returned an invalid explanation: {content}")
    exp_start_index = content.index("[EXPLANATION]:") + len("[EXPLANATION]:")
    return content[exp_start_index:].strip(), display_messages(system_prompt, messages)


"""
For convenience, here is a full output:
====================[[ SYSTEM PROMPT ]]====================
You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire sequence of tokens will be contained between delimiters <<just like this>>. How important each token is for the behavior is listed after each example in parentheses.

- Try to produce a concise final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Do not mention the marker tokens (<< >>) in your explanation.
- Do not make lists of possible explanations. Keep your explanations short and concise.
- The last line of your response must be the formatted explanation, using [EXPLANATION]:

====================[[ USER ]]====================

Example 1:  and he was <<over the moon>> to find
Activations: ("over", 5), (" the", 6), (" moon", 9)
Example 2:  we'll be laughing <<till the cows come home>>! Pro
Activations: ("till", 5), (" the", 5), (" cows", 8), (" come", 8), (" home", 8)
Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd
Activations: ("than", 5), (" meets", 7), (" the", 6), (" eye", 8)

====================[[ ASSISTANT ]]====================
[EXPLANATION]: Common idioms in text conveying positive sentiment.
====================[[ USER ]]====================

Example 1:  a river is wide but the ocean is wid<<er>>. The ocean
Activations: ("er", 8)
Example 2:  every year you get tall<<er>>," she
Activations: ("er", 2)
Example 3:  the hole was small<<er>> but deep<<er>> than the
Activations: ("er", 9), ("er", 9)

====================[[ ASSISTANT ]]====================
[EXPLANATION]: The token 'er' at the end of a comparative adjective describing size.
====================[[ USER ]]====================

Example 1:  something happening inside my <<house>>", he
Activations: ("house", 7)
Example 2:  presumably was always contained in <<a box>>", according
Activations: ("a", 5), ("box", 9)
Example 3:  people were coming into the <<smoking area>>".

However he
Activations: ("smoking", 2), ("area", 4)
Example 4:  Patrick: "why are you getting in the << way?>>" Later,
Activations: ("way", 4), ("?", 2)

====================[[ ASSISTANT ]]====================
[EXPLANATION]: Nouns representing a distinct objects that contains something, sometimes preciding a quotation mark.
====================[[ USER ]]====================
Example 1:   shorter and would>> probably<< be preferred>> by a native<< speaker.>>  <<It>> could<< be further shortened to

>>Despite playing well, we lost the game.

<<<bos>>>Q:


Activations: (" shorter", 17), (" and", 9), (" would", 4), (" be", 3), (" preferred", 3), (" speaker", 3), (".", 5), ("It", 7), (" be", 3), (" further", 5), (" shortened", 34), (" to", 18), ("

", 7), ("<bos>", 15)
Example 2:  }]}].
I'm sure it could be done<< better>> or more<< compact though,>> I don't have much experience with PHP in this regard.


Activations: (" better", 7), (" compact", 30), (" though", 4), (",", 11)
Example 3:  << chunks>> of DOM you can dynamically manipulate. The *
  <<syntax>> is<< a shortcut that lets you avoid>> writing the whole 
  element<<.>>

<ul class
Activations: (" chunks", 4), ("syntax", 3), (" a", 3), (" shortcut", 26), (" that", 11), (" lets", 10), (" you", 7), (" avoid", 5), (".", 19)
Example 4:  0185UJ0')]
return $x

which you can<< simplify to the>> XPath<< expression
>>//book[.//
Activations: (" simplify", 25), (" to", 16), (" the", 7), (" expression", 3), ("
", 3)
Example 5:   laundry detergent is>> a great example of how an impressive<< resource savings from>> a<< super-concentrated detergent>> directly leads to an easier and better user experience.

Fourth, establishing
Activations: (" laundry", 4), (" detergent", 5), (" is", 6), (" resource", 6), (" savings", 16), (" from", 5), (" super", 4), ("-", 4), ("concentr", 25), ("ated", 18), (" detergent", 6)
Example 6:   psycopg2 can somehow<< optimize commands that>> are executed many times<< but with>> different values<<,>> how and is it worth
Activations: (" optimize", 25), (" commands", 3), (" that", 10), (" but", 5), (" with", 4), (",", 6)
Example 7:   avoiding personal names, a preference for using recognition<<als>>, a preference for being<< succinct,>> and a pair of opposed preferences relating to referential specificity - guide speakers towards
Activations: ("als", 3), (" succinct", 24), (",", 4)
Example 8:   f.read()

A:

Let's start ref<<actoring>>/<<optimizations:>>

urllib should be<< replaced with>> requests<< library>> which is the de facto standard
Activations: ("actoring", 7), ("optim", 24), ("izations", 16), (":", 9), (" replaced", 4), (" with", 3), (" library", 3)

====================[[ FINAL ASSISTANT RESPONSE ]]====================
[EXPLANATION]: Words and phrases related to improving or making something more efficient, particularly in technical contexts. Common terms include "optimize", "shortened", "compact", "simplify", "concentrated", and "succinct".
"""
