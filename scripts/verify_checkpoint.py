import torch

"""
Verify checkpoint metadata is present
(should this be a unit test. yeah probably)
"""
def main():
    try:
        checkpoint = torch.load("models/lesson-05-pretrained/checkpoint.pt")
        print(f"epoch: {checkpoint["epoch"]}",
              f"\ntrain_losses: {checkpoint["train_losses"]}",
              f"\nval_losses: {checkpoint["val_losses"]}",
              f"\ntrack_tokens_seen: {checkpoint["track_tokens_seen"]}")
    except FileNotFoundError:
        print("No Checkpoint file found")

if __name__ == "__main__":
    main()
