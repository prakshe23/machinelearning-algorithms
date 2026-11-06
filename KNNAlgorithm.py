import os
import sys
import numpy as np

RAW_TRAINING_DATA = [
    # (Author,    Thread, Length, Where, Action)
    ("known",   "new", "long",  "home", "reads"),  # e1
    ("unknown", "new", "short", "work", "skips"),  # e2
    ("unknown", "old", "long",  "work", "skips"),  # e3
    ("known",   "new", "long",  "home", "skips"),  # e4
    ("known",   "old", "short", "work", "reads"),  # e5
    ("known",   "old", "long",  "home", "skips"),  # e6
]

# Maps each categorical value to +1 or -1
ENCODING = {
    "known": 1, "unknown": -1,
    "new": 1, "old": -1,
    "long": 1, "short": -1,
    "home": 1, "work": -1,
    "skips": 1, "reads": -1,
}

def write_data_file(path, raw_rows=RAW_TRAINING_DATA):
    with open(path, "w") as f:
        for row in raw_rows:
            encoded = [ENCODING[value] for value in row]
            f.write(" ".join(str(v) for v in encoded) + "\n")

RAW_TEST_SAMPLE = ("unknown", "new", "long", "home")
def write_test_file(path, raw_row=RAW_TEST_SAMPLE):
    with open(path, "w") as f:
        encoded = [ENCODING[value] for value in raw_row]
        f.write(" ".join(str(v) for v in encoded) + "\n")


def read_test_sample(path):
    with open(path, "r") as f:
        line = f.readline().strip()
    values = [float(v) for v in line.split()]
    return np.array(values)


def read_data(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = [float(v) for v in line.split()]
            rows.append(values)

    data = np.array(rows, dtype=float)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def distance(x1, x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return int(np.sum(x1 != x2))

def knn_predict(D, K, x_hat):
    X, y = D
    N = X.shape[0]

    S = []
    for n in range(N):
        d = distance(X[n], x_hat)
        S.append((d, n))

    S.sort(key=lambda pair: pair[0])

    y_hat = 0
    for k in range(K):
        dist, n = S[k]
        y_hat += y[n]

    if y_hat > 0:
        return 1
    elif y_hat < 0:
        return -1
    else:
        return 0


LABEL_NAMES = {1: "skips", -1: "reads"}


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = os.path.join(script_dir, "data.txt")
    test_path = os.path.join(script_dir, "test_data.txt")

    write_data_file(data_path)
    X_train, y_train = read_data(data_path)
    D = (X_train, y_train)

    write_test_file(test_path)
    x_test = read_test_sample(test_path)

    print("Training data (X):")
    print(X_train)
    print("Training labels (y):")
    print(y_train)
    print("Test sample e7 (features):", x_test)
    print()

    pred_k1 = knn_predict(D, 1, x_test)
    print("k = 1 prediction:", pred_k1, "->", LABEL_NAMES.get(pred_k1, "tie"))

    pred_k3 = knn_predict(D, 3, x_test)
    print("k = 3 prediction:", pred_k3, "->", LABEL_NAMES.get(pred_k3, "tie"))


if __name__ == "__main__":
    main()