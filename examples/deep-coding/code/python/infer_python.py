"""Run inference with ONNX Runtime on a few MNIST test samples."""
import numpy as np
import onnxruntime as ort
from torchvision import datasets, transforms


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tfm)

    n = 8
    batch = np.stack([test_ds[i][0].numpy() for i in range(n)]).astype(np.float32)
    labels = [test_ds[i][1] for i in range(n)]

    sess = ort.InferenceSession(
        "mnist_cnn.onnx",
        providers=["CPUExecutionProvider"],
    )
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    logits = sess.run([out_name], {in_name: batch})[0]
    probs = softmax(logits)
    preds = probs.argmax(1)

    for i in range(n):
        print(f"sample {i}: pred={preds[i]} (p={probs[i, preds[i]]:.3f}) "
              f"true={labels[i]}")
    acc = (preds == np.array(labels)).mean()
    print(f"Batch accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()