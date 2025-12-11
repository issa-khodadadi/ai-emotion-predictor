import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def main():
    texts = []
    labels = []

    # reading the data text file with ';' seperator
    with open("data/test.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ";" not in line:
                continue
            text, label = line.split(";", 1)
            texts.append(text.strip())
            labels.append(label.strip())

    print(f"Loaded {len(texts)} samples")

    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)
    num_labels = len(le.classes_)
    print("Classes:", le.classes_)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    dataset = EmotionDataset(texts, labels_enc, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
    )

    args = TrainingArguments(
        output_dir="model_output",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        save_steps=10_000,
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset
    )

    trainer.train()
    model.save_pretrained("model_output")
    tokenizer.save_pretrained("model_output")
    torch.save(le, "model_output/label_encoder.pt")
    print("Training finished. model saved in 'model_output/'.")


if __name__ == "__main__":
    main()
