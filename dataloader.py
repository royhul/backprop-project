#MNIST data loader
import gzip, os, urllib.request
import numpy as np

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test":  "t10k-images-idx3-ubyte.gz",
    "y_test":  "t10k-labels-idx1-ubyte.gz",
}

def _download(name, root):
    path = os.path.join(root, name)
    if not os.path.exists(path):
        print("downloading", name)
        urllib.request.urlretrieve(MIRROR + name, path)
    return path

def _read_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)   # 16-byte header
    return data.reshape(-1, 784)

def _read_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(np.int64)  # 8-byte header

def load_mnist(root="data/mnist"):
    """x_train (60000,784) float, y_train (60000,) int64, x_test (10000,784), y_test (10000,)."""
    os.makedirs(root, exist_ok=True)
    p = {k: _download(v, root) for k, v in FILES.items()}
    x_train, x_test = _read_images(p["x_train"]), _read_images(p["x_test"])
    y_train, y_test = _read_labels(p["y_train"]), _read_labels(p["y_test"])
    # scale to [0,1], then normalize with the standard MNIST train mean/std
    x_train = (x_train / 255.0 - 0.1307) / 0.3081
    x_test  = (x_test  / 255.0 - 0.1307) / 0.3081
    return x_train, y_train, x_test, y_test
