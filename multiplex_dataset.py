import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class BrainGraphDataset(Dataset):
    def __init__(self, n_samples=50, n_nodes=32, n_bands=5, n_layers=3, threshold=0.45):
        super().__init__()
        # Use a localized random seed for consistent data generation during debugging
        np.random.seed(1337)
        
        # Simulate power spectral features across nodes
        self.node_features = torch.randn(n_samples, n_nodes, n_bands)
        
        # Build raw random connectivity matrices
        raw_matrices = np.random.rand(n_samples, n_layers, n_nodes, n_nodes)
        
        # Apply the hard noise-gate limit specified in the pipeline rules
        raw_matrices[raw_matrices < threshold] = 0.0
        self.adjacency_tensors = torch.FloatTensor(raw_matrices)
        
        # Target state variables (binary tracking marker for detection classification)
        self.state_labels = torch.bernoulli(torch.full((n_samples, 1), 0.5))

    def __len__(self):
        return len(self.state_labels)

    def __getitem__(self, index):
        return self.node_features[index], self.adjacency_tensors[index], self.state_labels[index]

if __name__ == "__main__":
    # Test batch sampling locally
    ds = BrainGraphDataset(n_samples=10)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    
    feats, adjs, targets = next(iter(loader))
    print("--- Local Dataset Engine Check ---")
    print(f"Features:  {feats.shape}")
    print(f"Adjacency: {adjs.shape}")
    print(f"Targets:   {targets.shape}")
