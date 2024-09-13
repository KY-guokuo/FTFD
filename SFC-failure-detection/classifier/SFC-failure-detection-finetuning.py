import nltk
import numpy as np
from huggingface_hub import HfFolder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import torch
from data_loader import id2label, label2id, load_training_dataset

torch.cuda.empty_cache()
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


MODEL_ID = "google/flan-t5-base"
REPOSITORY_ID = "./SFC-failure-classification"

config = AutoConfig.from_pretrained(
    MODEL_ID, num_labels=len(label2id), id2label=id2label, label2id=label2id
)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, config=config)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# tokenizer.to("cuda") if torch.cuda.is_available() else tokenizer.to("cpu")

training_args = TrainingArguments(
    num_train_epochs=2,
    output_dir=REPOSITORY_ID,
    logging_strategy="steps",
    logging_steps=100,
    report_to="tensorboard",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    #fp16=True,
    fp16=False,  # Overflows with fp16
    learning_rate=1e-4,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,
    # push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=REPOSITORY_ID,
    hub_token=HfFolder.get_token(),
    #no_cuda=True,
)


def tokenize_function(examples) -> dict:
    """Tokenize the text column in the dataset"""
    # inputs = tokenizer(examples["text"], padding="max_length", truncation=True)
    # print(type(inputs))
    # inputs.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer(examples["text"], padding="max_length", truncation=True)
    # return inputs


def compute_metrics(eval_pred) -> dict:
    """Compute metrics for evaluation"""
    logits, labels = eval_pred
    if isinstance(
        logits, tuple
    ):  # if the model also returns hidden_states or attentions
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    report = classification_report(labels, predictions)
    print(report)
    cm = confusion_matrix(labels, predictions)
    print(cm)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def train() -> None:
    """
    Train the model and save it.
    """
    dataset = load_training_dataset("AutoModelForSequenceClassification")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # print(type(tokenized_datasets))
    nltk.download("punkt")
    # tokenized_datasets.to("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    # TRAIN
    # print(type(trainer))
    trainer.train()
    # SAVE AND EVALUATE
    tokenizer.save_pretrained(REPOSITORY_ID)
    trainer.create_model_card()
    torch.cuda.empty_cache()
    # trainer.push_to_hub()
    print(trainer.evaluate())


if __name__ == "__main__":
    train()
