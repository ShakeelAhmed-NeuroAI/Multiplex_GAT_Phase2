import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from multiplex_dataset import BrainGraphDataset

class ModelCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_source):
        super().__init__()
        self.dataset = data_source
        self.cursor = 0
        self.total_records = len(data_source)

    def get_next(self):
        if self.cursor >= self.total_records:
            return None
            
        features, adj, _ = self.dataset[self.cursor]
        self.cursor += 1
        
        # Format tensors with batch dimensions for tracing inputs
        return {
            "node_features": np.expand_dims(features.numpy(), axis=0),
            "multiplex_adj": np.expand_dims(adj.numpy(), axis=0)
        }

def convert_model_to_int8():
    print(">>> Initializing model quantization pass (FP32 -> INT8)...")
    
    cal_data = BrainGraphDataset(n_samples=20)
    reader = ModelCalibrationDataReader(cal_data)
    
    src_model = "multiplex_model.onnx"
    target_model = "multiplex_model_quantized.onnx"
    
    quantize_static(
        model_input=src_model,
        model_output=target_model,
        calibration_data_reader=reader,
        quant_format=QuantType.QInt8,
        per_channel=False
    )
    print(f">>> Quantization pass complete. Saved optimized file: '{target_model}'")

if __name__ == "__main__":
    convert_model_to_int8()
