# RQAE

RQAE (Residual Quantization AutoEncoder) is a model architecture to interpret LLMs. You can find more details [here](https://www.hkamath.me/blog/2024/rqae).

This repository consists of three main parts:
1. `rqae/`: The base model code that is portable and can be used in your own project.
    a. `model.py`: The RQAE model definition.
    b. `feature.py`: A single feature definition (what model it came from, what are its top activations and the intensities for those activations, etc.)
    c. `llm.py`: Adjusting a [transformers](https://www.github.com/huggingface/transformers) LLM to be used for interpretability methods, e.g. early stopping layers and adding forward hooks to save activations. Currently, it is only defined for Gemma2 models, but you can see that it's very straightforward to extend that (just need to change the `norm` and `denorm` functions).
    d. `gemmascope.py`: Gemmascope model definition.
2. `server/`: Code for the server and frontend of the demo. The server is hosted in Modal, so a lot of the code is written for Modal specifically.
3. `scripts/`: Scripts to prepare your own dataset for use in the demo code. Also includes scripts to run evals on features.
