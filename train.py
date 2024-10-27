import numpy as np
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

label_to_head = {0: "category", 1: "intensity", 2: "concern", 3: "polarity"}


def prep(ex):
    for label in ("Category", "Polarity", "Extracted Concern"):
        ex[label] = labels[label][ex[label]]
    return ex


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.texts = dataset["User Input"]
        self.labels = torch.tensor(
            [
                [
                    sample["Category"],
                    sample["Intensity"],
                    sample["Extracted Concern"],
                    sample["Polarity"],
                ]
                for sample in dataset
            ],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class MultiLabelClassifier(nn.Module):
    def __init__(self, model_name="all-mpnet-base-v2"):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        self.hidden_size = 768
        self.dropout = nn.Dropout(0.1)

        self.heads = nn.ModuleDict(
            {
                "category": ClassificationHead(self.hidden_size, 8),
                "intensity": ClassificationHead(self.hidden_size, 11),
                "concern": ClassificationHead(self.hidden_size, 11),
                "polarity": ClassificationHead(self.hidden_size, 3),
            }
        )

    def forward(self, sentences):
        embeddings = self.encoder.encode(sentences, convert_to_tensor=True)
        embeddings = self.dropout(embeddings)

        return {name: head(embeddings) for name, head in self.heads.items()}


def train_model(
    model, train_dataset, val_dataset, num_epochs=5, batch_size=16, learning_rate=2e-5
):
    wandb.init(
        project="mega",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": num_epochs,
        },
    )
    train_loader = DataLoader(
        CustomDataset(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device="cuda"),
    )
    val_loader = DataLoader(
        CustomDataset(val_dataset),
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator(device="cuda"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_val_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_texts, batch_labels in progress_bar:
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_texts)

            loss = sum(
                criterion(outputs[head], batch_labels[:, i])
                for i, head in label_to_head.items()
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"training_loss": "{:.3f}".format(loss.item())})
            wandb.log({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        correct_predictions = {head: 0 for head in model.heads.keys()}
        total_predictions = 0

        with torch.no_grad():
            for batch_texts, batch_labels in val_loader:
                batch_labels = batch_labels.to(device)
                outputs = model(batch_texts)

                total_predictions += batch_labels.size(0)

                for i, head in label_to_head.items():
                    _, predicted = torch.max(outputs[head], 1)
                    correct_predictions[head] += (
                        (predicted == batch_labels[:, i]).sum().item()
                    )

        print(f"\nEpoch {epoch + 1}")
        print(f"Average training loss: {avg_train_loss:.3f}")
        print("Validation Accuracies:")
        accuracies = {}
        for head in model.heads.keys():
            accuracy = correct_predictions[head] / total_predictions
            accuracies[head] = accuracy
            print(f"{head.capitalize()}: {accuracy:.3f}")

        avg_accuracy = sum(accuracies.values()) / len(accuracies)
        if avg_accuracy > best_val_accuracy:
            best_val_accuracy = avg_accuracy
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved best model!")

    wandb.finish()


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    torch.set_default_device("cuda")

    dataset = load_dataset("csv", data_files="sheet.tsv", delimiter="\t")
    labels = {
        label: {k: i for i, k in enumerate(set(dataset["train"][label]))}
        for label in ("Category", "Polarity", "Extracted Concern")
    }
    invert_labels = {
        label: {i: k for i, k in enumerate(set(dataset["train"][label]))}
        for label in ("Category", "Polarity", "Extracted Concern")
    }
    dataset["train"] = dataset["train"].map(prep)

    train_testvalid = dataset["train"].train_test_split(test_size=0.2)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5)

    dataset = {
        "train": train_testvalid["train"],
        "val": test_valid["train"],
        "test": test_valid["test"],
    }
    model = MultiLabelClassifier()
    train_model(
        model,
        train_dataset=dataset["train"],
        val_dataset=dataset["val"],
        num_epochs=5,
        batch_size=100,
        learning_rate=2e-3,
    )
