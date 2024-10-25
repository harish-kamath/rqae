# Taken from https://arxiv.org/abs/2410.13928 (Appendix A.1) and https://github.com/EleutherAI/sae-auto-interp/blob/8ecfc02986f61595c53dd1befa15c0cd88cf3ec8/sae_auto_interp/explainers/default/prompts.py#L3

system_prompt = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire sequence of tokens will be contained between delimiters <<just like this>>. How important each token is for the behavior is listed after each example in parentheses.

- Try to produce a concise final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Do not mention the marker tokens (<< >>) in your explanation.
- Do not make lists of possible explanations. Keep your explanations short and concise.
- The last line of your response must be the formatted explanation, using [EXPLANATION]:
"""

example_inputs = [
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

example_outputs = [
    "[EXPLANATION]: Common idioms in text conveying positive sentiment.",
    "[EXPLANATION]: The token 'er' at the end of a comparative adjective describing size.",
    "[EXPLANATION]: Nouns representing a distinct objects that contains something, sometimes preciding a quotation mark.",
]
