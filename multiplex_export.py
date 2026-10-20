import torch
from multiplex_model import MultiplexBrainEngine
from multiplex_dataset import BrainGraphDataset

def serialize_to_onnx():
    print(">>> Initializing graph model deployment serialization...")
    
    # Initialize evaluation engine layout
    net = MultiplexBrainEngine()
    net.eval()
    
    # Create sample verification traces matching edge hardware dimensions
    mock_x = torch.randn(1, 32, 5)
    mock_adj = torch.rand(1, 3, 32, 32)
    
    save_path = "multiplex_model.onnx"
    
    # Export execution flow graph definitions
    torch.onnx.export(
        net,
        (mock_x, mock_adj),
        save_path,
        input_names=["node_features", "multiplex_adj"],
        output_names=["state_confidence"],
        opset_version=17,
        dynamic_axes={
            "node_features": {0: "batch_size"},
            "multiplex_adj": {0: "batch_size"},
            "state_confidence": {0: "batch_size"}
        }
    )
    print(f">>> Successfully exported compiled binary file to: '{save_path}'")

if __name__ == "__main__":
    serialize_to_onnx()
