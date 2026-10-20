import time
import numpy as np
import onnxruntime as ort

def run_hardware_benchmarking():
    print(">>> Running hardware engine speed checks...")
    
    # Configure optimization runtime execution options
    runtime_opts = ort.SessionOptions()
    runtime_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    engine = ort.InferenceSession("multiplex_model_quantized.onnx", runtime_opts, providers=['CPUExecutionProvider'])
    
    # Get layer bindings
    feat_key = engine.get_inputs()[0].name
    adj_key = engine.get_inputs()[1].name
    
    # Sample tensors matching configuration matrices shape
    feat_feed = np.random.randn(1, 32, 5).astype(np.float32)
    adj_feed = np.random.rand(1, 3, 32, 32).astype(np.float32)
    adj_feed[adj_feed < 0.45] = 0.0
    
    data_map = {feat_key: feat_feed, adj_key: adj_feed}
    
    # Cache execution warmups
    for _ in range(10):
        _ = engine.run(None, data_map)
        
    # Timed benchmarking loops
    records = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = engine.run(None, data_map)
        t1 = time.perf_counter()
        records.append((t1 - t0) * 1000.0) # Map directly to ms
        
    print("\n================ BENCHMARK REPORT ================")
    print(f" Average Latency   : {np.mean(records):.4f} ms")
    print(f" Jitter (Std Dev)  : {np.std(records):.4f} ms")
    print(f" Maximum Latency   : {np.max(records):.4f} ms")
    print(f" Minimum Latency   : {np.min(records):.4f} ms")
    print("==================================================")

if __name__ == "__main__":
    run_hardware_benchmarking()
