from time import time
from typing import List, Tuple

import torch
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# from classifier.data_loader import id2label, load_dataset
from data_loader_new import id2label, load_test_dataset
dataset = load_test_dataset("AutoModelForSequenceClassification")

# Load the model and tokenizer. MODEL_DIR is the model directory, and TOKENIZER_ID is the tokenizer directory.
MODEL_DIR = "./SFC-failure-classification/checkpoint-18312"
TOKENIZER_ID = "./SFC-failure-classification"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to("cuda") if torch.cuda.is_available() else model.to("cpu")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)


def classify(texts_to_classify: List[str]) -> List[Tuple[str, float]]:
    """Classify a list of failure using the model."""

    # Tokenize all texts in the batch
    start = time()
    inputs = tokenizer(
        texts_to_classify,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )
    print(type(inputs))
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    logger.debug(
        f"Classification of {len(texts_to_classify)} examples took {time() - start} seconds"
    )

    # Process the outputs to get the probability distribution
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get the top class and the corresponding probability (certainty) for each input text
    confidences, predicted_classes = torch.max(probs, dim=1)
    predicted_classes = (
        predicted_classes.cpu().numpy()
    )
    # Move to CPU for numpy conversion
    confidences = confidences.cpu().numpy()

    # Map predicted class IDs to labels
    predicted_labels = [id2label[class_id] for class_id in predicted_classes]

    # Zip together the predicted labels and confidences and convert to a list of tuples
    return list(zip(predicted_labels, confidences))


def evaluate():
    """Evaluate the model on the test dataset."""
    predictions_list, labels_list = [], []

    batch_size = 16  # Adjust batch size based GPU capacity
    num_batches = len(dataset) // batch_size + (
        0 if len(dataset) % batch_size == 0 else 1
    )
    progress_bar = tqdm(total=num_batches, desc="Evaluating")
    for i in range(0, len(dataset), batch_size):
        batch_texts = dataset["text"][i : i + batch_size]
        batch_labels = dataset["label"][i : i + batch_size]
        batch_predictions = classify(batch_texts)
        predictions_list.extend(batch_predictions)
        labels_list.extend([id2label[label_id] for label_id in batch_labels])
        progress_bar.update(1)

    progress_bar.close()
    predictions = [pair[0] for pair in predictions_list]
    accuracy = accuracy_score(labels_list, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_list, predictions, average="weighted"
    )
    print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1:{f1}")
    report = classification_report(labels_list, [pair[0] for pair in predictions_list])
    print(report)
    cm = confusion_matrix(labels_list, predictions)
    print(cm)


if __name__ == "__main__":
    evaluate()
