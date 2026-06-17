"""Export the trained CNN to ONNX (opset 17)."""
import torch
from train import SimpleCNN


def main():
    model = SimpleCNN()
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 1, 28, 28)

    torch.onnx.export(
        model,
        dummy,
        "mnist_cnn.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print("Exported -> mnist_cnn.onnx")

    import onnx
    onnx.checker.check_model(onnx.load("mnist_cnn.onnx"))
    print("ONNX model is valid.")


if __name__ == "__main__":
    main()