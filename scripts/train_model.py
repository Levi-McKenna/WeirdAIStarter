import torch
from torch.utils.data import DataLoader

from weird_ai.dataset import LyricsDataset
from weird_ai.model import WeirdAIModel
from weird_ai.tokenizer import SimpleCharacterTokenizer
from weird_ai.trainer import save_checkpoint, load_checkpoint, train_model_simple
from weird_ai.config import SAMPLE_LYRICS_FILE, WEIRD_AI_CFG, PROJECT_ROOT


def create_data_loader(tokenizer, text, batch_size, max_length, stride, shuffle=True,
                       drop_last=True, num_workers=0):
    dataset = LyricsDataset(tokenizer.encode(text), batch_size, stride)
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle, drop_last=drop_last,
        num_workers=num_workers
    )


TRAIN_SETTINGS = {
    "learning_rate": 0.0001,
    "weight_decay": .01,
    "batch_size": 12,
    "num_epoch": 1,
    "eval_freq": 6,
    "prompt": "Love is ",
    "checkpoint_path": PROJECT_ROOT / "models/lesson-05-pretrained/checkpoint.pt"
}

def main():
    print("Starting training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = SAMPLE_LYRICS_FILE.read_text(encoding="utf-8")[:100000]
    tokenizer = SimpleCharacterTokenizer(text)

    vocab_size = len(tokenizer.stoi)
    model = WeirdAIModel(vocab_size, WEIRD_AI_CFG["context_length"],
                         WEIRD_AI_CFG["emb_dim"], WEIRD_AI_CFG["n_layers"],
                         WEIRD_AI_CFG["qkv_bias"])
    # TODO:
    # 1. Load training data
    # 2. Create optimizer
    # 3. Implement training loop
    # 4. Save checkpoints
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_SETTINGS["learning_rate"],
        weight_decay=TRAIN_SETTINGS["weight_decay"]
    )

    checkpoint = None
    try:
        checkpoint = load_checkpoint(model, optimizer, TRAIN_SETTINGS["checkpoint_path"], device)
    except FileNotFoundError:
        print("No Checkpoint file found")

    # loaders:
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text))

    train_loader = create_data_loader(
        tokenizer,
        text[:split_idx],
        batch_size=TRAIN_SETTINGS["batch_size"],
        max_length=WEIRD_AI_CFG["context_length"],
        stride=WEIRD_AI_CFG["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_data_loader(
        tokenizer,
        text[split_idx:],
        batch_size=TRAIN_SETTINGS["batch_size"],
        max_length=WEIRD_AI_CFG["context_length"],
        stride=WEIRD_AI_CFG["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device,
                       TRAIN_SETTINGS["num_epoch"], TRAIN_SETTINGS["eval_freq"], TRAIN_SETTINGS["prompt"],
                       tokenizer, WEIRD_AI_CFG["context_length"])
    epoch = TRAIN_SETTINGS["num_epoch"]

    if checkpoint is not None:
        epoch += checkpoint["epoch"]
        tokens_seen = checkpoint["track_tokens_seen"] + tokens_seen
        train_losses = checkpoint["train_losses"] + train_losses
        val_losses = checkpoint["val_losses"] + val_losses

    save_checkpoint(model, optimizer, epoch, train_losses,
                    val_losses, tokens_seen, TRAIN_SETTINGS["checkpoint_path"])
    print("Training complete.")


if __name__ == "__main__":
    main()
