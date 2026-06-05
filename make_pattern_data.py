import numpy as np

N = 6000
D = 768
C = 3

SEG = 1000

features = np.zeros((N, D), dtype=np.float32)
labels = np.zeros((N, C), dtype=np.float32)

for i in range(N):

    phase = (i // SEG) % 3

    if phase == 0:

        features[i] = (
            np.random.randn(D) + 2
        )

        labels[i, 0] = 1

    elif phase == 1:

        features[i] = (
            np.random.randn(D)
        )

        labels[i, 1] = 1

    else:

        features[i] = (
            np.random.randn(D) - 2
        )

        labels[i, 2] = 1

np.save(
    "data/features.npy",
    features
)

np.save(
    "data/labels.npy",
    labels
)

print("saved")