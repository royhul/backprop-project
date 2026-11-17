from dataloader import load_mnist
from model import MLP
from optimiser import SGD
from operations import SoftmaxCrossEntropy
import numpy as np

def accuracy(logits, labels):
    if hasattr(logits, "data"): 
        logits = logits.data
    pred = np.argmax(logits, axis=1)
    return (pred == labels).mean()

def train():
    x_train, y_train, x_test, y_test = load_mnist()
    rng = np.random.default_rng(0)               # fixed seed, same split every run
    perm = rng.permutation(len(x_train))
    val_idx, tr_idx = perm[:5000], perm[5000:]
    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[tr_idx], y_train[tr_idx]
    model = MLP(784, 256, 128, 10)
    params = model.parameters()
    optim = SGD(params, lr=0.1)
    
    for epoch in range(20):
        idx = np.random.permutation(len(x_train))
        count = 0
        train_loss = 0
        for start in range(0, len(x_train), 64):        
            batch_idx = idx[start:start + 64]
            optim.zero_grad()
            output = model(x_train[batch_idx])
            loss = SoftmaxCrossEntropy.run(output, labels=y_train[batch_idx])
            train_loss += loss.data
            count += 1
            loss.backward()
            optim.step()
        train_loss = train_loss/count
        val_loss =  SoftmaxCrossEntropy.run(model(x_val), labels=y_val)
        print(f"Epoch No:{epoch} -> Train_Loss:{train_loss} Val Loss:{val_loss.data}")

    acc = accuracy(model(x_test), y_test)
    print(f"Final Test accuracy:{(acc*100)}")


if __name__ == "__main__":
    train()