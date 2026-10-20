import numpy as np
from scipy.signal import welch

class MultiplexGraphExtractor:
    def __init__(self, num_channels=32, fs=250, threshold=0.45):
        self.num_channels = num_channels
        self.fs = fs
        self.threshold = threshold
        self.bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 12.0),
            'beta':  (12.0, 30.0),
            'gamma': (30.0, 45.0)
        }

    def _extract_band_powers(self, Z_k):
        freqs, psd = welch(Z_k, fs=self.fs, nperseg=125, axis=-1)
        band_features = []
        for band_name in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            f_min, f_max = self.bands[band_name]
            idx = np.where((freqs >= f_min) & (freqs <= f_max))[0]
            mean_power = np.mean(psd[:, idx], axis=-1) 
            band_features.append(mean_power)
        return np.array(band_features) 

    def compute_multiplex_tensor(self, Z_k):
        bands_power = self._extract_band_powers(Z_k)
        multiplex_tensor = np.zeros((3, self.num_channels, self.num_channels), dtype=np.float32)
        layer_mappings = [(1, 3), (2, 4), (0, 2)]
        
        for layer_idx, (b1, b2) in enumerate(layer_mappings):
            vec1 = bands_power[b1] 
            vec2 = bands_power[b2] 
            outer_matrix = np.outer(vec1, vec2)
            
            # Proper scaling per layer
            mat_min = np.min(outer_matrix)
            mat_max = np.max(outer_matrix)
            if (mat_max - mat_min) > 1e-6:
                outer_matrix = (outer_matrix - mat_min) / (mat_max - mat_min)
                
            outer_matrix[np.abs(outer_matrix) < self.threshold] = 0.0
            multiplex_tensor[layer_idx] = outer_matrix
            
        return multiplex_tensor

if __name__ == "__main__":
    print("Initializing Multiplex Cross-Frequency Graph Extractor...")
    extractor = MultiplexGraphExtractor(num_channels=32, fs=250, threshold=0.45)
    
    np.random.seed(42)
    simulated_Z_k = np.random.normal(loc=0.0, scale=1.0, size=(32, 250))
    tensor = extractor.compute_multiplex_tensor(simulated_Z_k)
    
    print("\n[SUCCESS] Extraction Pipeline Executed.")
    print(f"Output Tensor Shape: {tensor.shape} (Layers x Channels x Channels)")
    print(f"Layer 0 (Theta-Beta) Active Edges: {np.count_nonzero(tensor[0])}")
    print(f"Layer 1 (Alpha-Gamma) Active Edges: {np.count_nonzero(tensor[1])}")
    print(f"Layer 2 (Delta-Alpha) Active Edges: {np.count_nonzero(tensor[2])}")
