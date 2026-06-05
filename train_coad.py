import torch
import torch.nn as nn

from utils.orthogonal_grad import OrthogonalGradient
from trainer import (
    load_samples, make_model, set_seed,
    train_epoch, evaluate,
    TRAIN_FEAT_DIR, VAL_FEAT_DIR,
)

SEED       = 0
NUM_EPOCHS = 10

set_seed(SEED)
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = make_model(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
orth      = OrthogonalGradient()

train_samples = load_samples("data/subset/train_mini.json")
val_samples   = load_samples("data/subset/val_mini.json")

print(f"[COAD] seed={SEED}  train={len(train_samples)}  val={len(val_samples)}")
print(f"{'epoch':>5}  {'train_loss':>10}  {'train_acc':>9}  {'val_loss':>8}  {'val_acc':>7}")

for epoch in range(NUM_EPOCHS):
    tr_loss, tr_acc = train_epoch(
        model, train_samples, TRAIN_FEAT_DIR, criterion, optimizer, device,
        orth=orth,
    )
    vl_loss, vl_acc = evaluate(
        model, val_samples, VAL_FEAT_DIR, criterion, device
    )
    print(
        f"{epoch+1:>5}  {tr_loss:>10.4f}  {tr_acc:>9.4f}"
        f"  {vl_loss:>8.4f}  {vl_acc:>7.4f}"
    )