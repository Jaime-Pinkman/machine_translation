from pathlib import Path

from clearml import Task
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from text_classifier.data.text_processing import (
    Tokenizer,
    Vocabulary,
    VectorizerFactory,
)
from text_classifier.data.datasets import SparseDatasetFactory, DenseDatasetFactory
from text_classifier.utils import load_params

BASE_DIR = Path().resolve()
PACKAGE = "src/text_classifier"
DATA = "data"
MODELS = "models"
INTERIM = "interim"
EXPS = "experiments"
CONFIG = "config"
LOGS = "logs"


def check_loss_accuracy(loader, model, criterion):
    num_correct = 0
    num_samples = 0
    loss = 0.0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)

            _, predictions = scores.max(1)

            loss += criterion(scores, y)

            num_correct += (predictions == y).sum()

            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples, loss / len(loader)


def train_eval_model(
    model,
    loaders,
    num_epochs,
    criterion,
    optimizer,
    device="cpu",
    save_model_path=None,
    writer=None,
):
    for epoch in tqdm(range(num_epochs)):
        for batch in loaders["train"]:
            data, targets = batch
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        for stage, loader in loaders.items():
            acc, loss = check_loss_accuracy(loader, model, criterion)
            if writer is not None:
                writer.add_scalar(f"Acc/{stage}", acc, epoch)
                writer.add_scalar(f"Loss/{stage}", loss, epoch)
            else:
                print(f"Acc/{stage}", acc.item(), epoch)
                print(f"Loss/{stage}", loss.item(), epoch)

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)


if __name__ == "__main__":

    EXP = "exp01"
    params = load_params(
        str(BASE_DIR / PACKAGE / CONFIG / EXPS / f"{EXP}_{CONFIG}.yaml")
    )
    learning_rate = params["lr"]
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    task = Task.init(
        project_name="News Classification",
        task_name="_".join([f"{key}{value}" for key, value in params.items()]),
        output_uri=True,
    )
    task.set_parameters(params)
    writer = SummaryWriter(log_dir=str(BASE_DIR / EXPS / EXP / LOGS))
    MAX_DF = 0.8
    MIN_COUNT = 5
    MIN_TOKEN_SIZE = 4

    vocabulary = Vocabulary(max_doc_freq=MAX_DF, min_count=MIN_COUNT)
    tokenizer = Tokenizer(min_token_size=MIN_TOKEN_SIZE)
    df_train = pd.read_csv(str(BASE_DIR / DATA / INTERIM / "train.csv"))
    df_test = pd.read_csv(str(BASE_DIR / DATA / INTERIM / "test.csv"))

    tokenized_texts_train = tokenizer.tokenize_corpus(list(df_train["text"]))
    tokenized_texts_test = tokenizer.tokenize_corpus(list(df_test["text"]))

    vocabulary.build(tokenized_texts_train)
    use_sparse = True
    vectorizer_factory = VectorizerFactory(
        vocabulary, mode="tfidf", scale="minmax", use_sparse=use_sparse
    )
    vectorizer = vectorizer_factory.get_vectorizer()
    train_vectors = vectorizer.vectorize(tokenized_texts_train)
    test_vectors = vectorizer.vectorize(tokenized_texts_test)

    train_targets = df_train["label_index"].to_numpy()
    test_targets = df_test["label_index"].to_numpy()
    dataset_factory = SparseDatasetFactory() if use_sparse else DenseDatasetFactory()

    main_dataset = dataset_factory.create_dataset(train_vectors, train_targets)
    test_dataset = dataset_factory.create_dataset(test_vectors, test_targets)
    train_dataset, val_dataset = random_split(main_dataset, [0.8, 0.2])
    input_size = len(vocabulary)
    num_classes = len(set(train_targets))

    loaders = {
        "train": DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False),
    }
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model = nn.Linear(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    train_eval_model(
        model=model,
        loaders=loaders,
        num_epochs=num_epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_model_path=str(BASE_DIR / EXPS / EXP / MODELS / "linear.pt"),
        writer=writer,
    )
    writer.flush()
