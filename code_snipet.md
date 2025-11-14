# Question Answering Model -- Full Code & Explanation

This Markdown file contains the **full code snippet** you provided,
along with explanations of how the model is trained using HuggingFace
Transformers.

------------------------------------------------------------------------

## 🚀 Complete Code Snippet

``` python
# STEP 1: Install dependencies & imports
!pip install -q torch torchvision torchaudio transformers datasets evaluate

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
    pipeline
)
from datasets import Dataset
import evaluate
import numpy as np
import os

# Disable Weights & Biases (no API key needed)
os.environ["WANDB_DISABLED"] = "true"

# ✅ STEP 2: Create Custom Dataset (10 QA pairs)
data = {
    "id": [str(i) for i in range(10)],
    "context": [
        "Natural Language Processing (NLP) is a field of Artificial Intelligence that focuses on the interaction between computers and humans through language.",
        "Tokenization is the process of breaking text into smaller units called tokens, which could be words, subwords, or characters.",
        "Stemming is a text normalization technique that reduces words to their base or root form.",
        "Lemmatization is similar to stemming but ensures that the reduced form of the word is a valid word in the language.",
        "A Transformer model uses attention mechanisms to process words in parallel, unlike RNNs that process sequentially.",
        "BERT stands for Bidirectional Encoder Representations from Transformers, a model that understands context from both directions.",
        "The attention mechanism helps models focus on relevant parts of input sentences when generating outputs.",
        "Word embeddings are dense vector representations of words that capture their semantic meaning.",
        "Fine-tuning involves taking a pre-trained model and training it further on a specific downstream task.",
        "Sequence-to-sequence models are used for tasks like translation, where one sequence is transformed into another."
    ],
    "question": [
        "What does NLP focus on?",
        "What is tokenization in NLP?",
        "What is stemming used for?",
        "How does lemmatization differ from stemming?",
        "How do Transformer models process words?",
        "What does BERT stand for?",
        "What is the purpose of attention mechanism?",
        "What are word embeddings?",
        "What does fine-tuning a model mean?",
        "What are sequence-to-sequence models used for?"
    ],
    "answers": [
        {"text": ["the interaction between computers and humans through language"], "answer_start": [86]},
        {"text": ["breaking text into smaller units called tokens"], "answer_start": [17]},
        {"text": ["reduces words to their base or root form"], "answer_start": [53]},
        {"text": ["ensures that the reduced form of the word is a valid word"], "answer_start": [49]},
        {"text": ["uses attention mechanisms to process words in parallel"], "answer_start": [17]},
        {"text": ["Bidirectional Encoder Representations from Transformers"], "answer_start": [10]},
        {"text": ["focus on relevant parts of input sentences"], "answer_start": [24]},
        {"text": ["dense vector representations of words that capture their semantic meaning"], "answer_start": [21]},
        {"text": ["taking a pre-trained model and training it further on a specific downstream task"], "answer_start": [21]},
        {"text": ["tasks like translation, where one sequence is transformed into another"], "answer_start": [41]},
    ]
}

# Convert to Dataset object
dataset = Dataset.from_dict(data)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# STEP 3: Tokenization
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=256,
        stride=32,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):
        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = inputs.sequence_ids(i)
        sample_idx = sample_map[i]
        answers = dataset["train"][sample_idx]["answers"]
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        context_index = 1

        token_start_index = 0
        while sequence_ids[token_start_index] != context_index:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != context_index:
            token_end_index -= 1

        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# STEP 4: Load Model
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# STEP 5: Training Arguments
args = TrainingArguments(
    output_dir="./qa_model",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir='./logs',
    logging_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# STEP 6: Train the model
trainer.train()

# STEP 7: Simple QA Prediction Function
def answer_question(context, question):
    device = next(model.parameters()).device
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)
        answer_tokens = inputs["input_ids"][0][start_idx:end_idx+1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer

# STEP 8: Demo Predictions
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad",
    tokenizer="distilbert-base-uncased-distilled-squad"
)

context = "Natural Language Processing (NLP) enables computers to understand human language."
question = "What does NLP enable computers to do?"
result = qa_pipeline(question=question, context=context)

print("Q:", question)
print("A:", result['answer'])

context2 = "Tokenization is the process of breaking text into smaller units called tokens, which could be words, subwords, or characters."
question2 = "What is tokenization in NLP?"
result2 = qa_pipeline(question=question, context=context2)

print("Q:", question2)
print("A:", result2['answer'])
```

------------------------------------------------------------------------
