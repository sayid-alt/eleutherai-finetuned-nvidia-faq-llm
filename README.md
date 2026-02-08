# EleutherAI Pythia-1B Fine-Tuned on NVIDIA FAQ

## Project Overview
This project fine-tunes the EleutherAI/pythia-1b language model on a curated NVIDIA documentation FAQ dataset to produce domain-aware answers about CUDA, GPUs, drivers, and related NVIDIA topics. It uses Hugging Face Transformers and the Trainer API, with optional logging to Weights & Biases. A simple Gradio interface is provided for interactive inference.

## Repository Structure
- `training/`: end-to-end notebook for data prep, training, and inference. See [training_notebook.ipynb](file:///Users/heykalsayid/Desktop/myown/llm/eleutherai-finetuned-nvidia-faq-llm/training/training_notebook.ipynb).
- `datasets/`: local copies of the source and auxiliary datasets, including:
  - [NvidiaDocumentationQandApairs.csv](file:///Users/heykalsayid/Desktop/myown/llm/eleutherai-finetuned-nvidia-faq-llm/datasets/NvidiaDocumentationQandApairs.csv)
  - [NvidiaDocumentationQandApairs.zip](file:///Users/heykalsayid/Desktop/myown/llm/eleutherai-finetuned-nvidia-faq-llm/datasets/NvidiaDocumentationQandApairs.zip)
  - [DataScienceBasics_QandA.csv](file:///Users/heykalsayid/Desktop/myown/llm/eleutherai-finetuned-nvidia-faq-llm/datasets/DataScienceBasics_QandA.csv)
- `requirements.txt`: runtime dependencies.

## Model and Dataset
- Base model: `EleutherAI/pythia-1b` (causal LM).
- Target dataset: NVIDIA Documentation Q&A pairs, formatted into prompt-completion pairs:
  ```
  ### Question:
  {question}

  ### Answer:
  {answer}
  ```
- A tokenized dataset is pushed to the Hugging Face Hub as `paacamo/nvidia-faq-eleutherai-fine-tuned` for reproducible training and inference.

## Training Configuration
Training is implemented with the Hugging Face `Trainer`. Key parameters (see the notebook for full details):
- epochs: 2
- learning rate: 2e-5
- optimizer: Adafactor
- per-device batch size: 8 (train and eval)
- eval/save/log every 100 steps
- gradient accumulation: 1
- gradient checkpointing: disabled
- early stopping-style selection via `load_best_model_at_end=True` with `metric_for_best_model="eval_loss"`
- push to hub: enabled (`push_to_hub=True`)

Refer to the exact configuration in [training_notebook.ipynb](file:///Users/heykalsayid/Desktop/myown/llm/eleutherai-finetuned-nvidia-faq-llm/training/training_notebook.ipynb#L592-L643).

## Fine-Tuning Results
- The best checkpoint is selected using evaluation loss (`eval_loss`) on the held-out split.
- Qualitatively, the fine-tuned model provides more precise answers to NVIDIA- and CUDA-specific questions compared to the base model.
- Example generation (using the Hub-hosted model):
  ```python
  from transformers import pipeline
  model_id = "paacamo/EleutherAI-pythia-1b-finetuned-nvidia-faq"
  pipe = pipeline("text-generation", model=model_id)
  print(pipe("What is the purpose of using CUDA rather than CPU?")[0]["generated_text"])
  ```
For a notebook-based inference example, see the inference cells in [training_notebook.ipynb](file:///Users/heykalsayid/Desktop/myown/llm/eleutherai-finetuned-nvidia-faq-llm/training/training_notebook.ipynb#L869-L906).

## Quick Start
1. Create a virtual environment (recommended) and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Gradio demo (from the notebook) to chat with the model. An inline example interface is provided in the notebook cells (Gradio install and `iface.launch(...)`).
3. Or use the Transformers pipeline directly (see the example above).

## Reproducing Training
Training is captured in the notebook:
- Data loading and prompt formatting
- Tokenization and splitting
- Trainer setup and execution
- Pushing dataset and model to the Hub

Open the notebook: [training_notebook.ipynb](file:///Users/heykalsayid/Desktop/myown/llm/eleutherai-finetuned-nvidia-faq-llm/training/training_notebook.ipynb). It is configured to run in GPU environments (e.g., Kaggle/Colab). Optional Weights & Biases logging can be enabled via environment variables set in the notebook cells.

## Notes
- Ensure Hugging Face authentication is set if you plan to push to/pull from private repos on the Hub.
- The model is specialized for NVIDIA documentation-style questions; general-purpose performance is not the focus.
