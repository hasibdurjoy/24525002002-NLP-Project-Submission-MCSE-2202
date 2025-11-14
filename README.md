# Fine-Tuning a Question Answering Model with Transformers

This guide explains how the provided Python code fine‑tunes a QA model
using **Hugging Face Transformers**, **PyTorch**, and a custom dataset.
Below is a clear walkthrough of each step.

------------------------------------------------------------------------

## 🧩 Step 1: Install Dependencies & Import Libraries

The code installs required libraries such as: - `transformers` → for
models, tokenizers, Trainer API\
- `datasets` → for handling dataset\
- `evaluate` → for future evaluation\
- `torch` → for deep learning\
- `os` → for environment variables

`WANDB_DISABLED` is set to prevent Weights & Biases from activating.

------------------------------------------------------------------------

## 🗂 Step 2: Creating a Custom Dataset

A dictionary of **10 QA pairs** is created with: - `context`\
- `question`\
- `answers` (with text + starting character position)

The dataset is converted into a HuggingFace `Dataset` and split:

``` python
dataset = dataset.train_test_split(test_size=0.2)
```

------------------------------------------------------------------------

## 🔤 Step 3: Tokenization & Preprocessing

**Tokenizer:** `distilbert-base-uncased`

### Key Tasks:

-   Combine `question + context`
-   Truncate context when long
-   Generate:
    -   `input_ids`
    -   `attention_mask`
    -   Overflow mapping (for long contexts)
    -   Start and end token positions for answers

### Why offset mapping?

Offset mapping links token positions back to the original string so we
can find:

    start_positions
    end_positions

This is required for training QA models.

------------------------------------------------------------------------

## 🤖 Step 4: Load the Base Model

``` python
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
```

DistilBERT is used because it is: - Lightweight\
- Fast\
- Great for small datasets

------------------------------------------------------------------------

## ⚙️ Step 5: Define Training Arguments

`TrainingArguments` configures: - Batch size\
- Learning rate\
- Epoch count\
- Output directory\
- Logging

The `Trainer` API simplifies: - Training\
- Evaluation\
- Saving the model

------------------------------------------------------------------------

## 🏋️ Step 6: Train the Model

``` python
trainer.train()
```

This fine‑tunes the model on your custom dataset.

------------------------------------------------------------------------

## ❓ Step 7: Build Simple QA Prediction Function

A custom function: - Tokenizes new question + context\
- Runs model inference\
- Extracts best start and end token\
- Decodes the answer

This allows manual testing outside the pipeline.

------------------------------------------------------------------------

## 🧪 Step 8: Demo with Pretrained Pipeline

HuggingFace pipeline is used as a comparison:

``` python
qa_pipeline = pipeline("question-answering",
                       model="distilbert-base-uncased-distilled-squad")
```

This is not your trained model---it's for demonstration.

------------------------------------------------------------------------

# 📝 Summary

This code demonstrates:

### ✔️ Creating and formatting a QA dataset

### ✔️ Preprocessing with tokenization and offset mapping

### ✔️ Fine‑tuning a DistilBERT QA model

### ✔️ Building a manual inference function

### ✔️ Testing with HuggingFace pipeline

It is a complete mini‑NLP project showing how to train your own QA model
from scratch.
