import numpy as np

N = 5000

features = np.random.randn(
    N,
    768
).astype(np.float32)

labels = np.random.randint(
    0,
    2,
    size=(N, 87)
).astype(np.float32)

np.save(
    "data/features.npy",
    features
)

np.save(
    "data/labels.npy",
    labels
)