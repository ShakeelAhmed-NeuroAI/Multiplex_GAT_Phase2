import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from multiplex_model import MultiplexBrainEngine
from multiplex_dataset import BrainGraphDataset

def run_local_training():
    print(">>> Initializing local network optimization sequence...")
    
    # Setup data loaders
    train_ds = BrainGraphDataset(n_samples=50)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    
    # Initialize model structures
    net = MultiplexBrainEngine()
    net.train()
    
    loss_fn = nn.BCELoss()
    opt = optim.Adam(net.parameters(), lr=0.005)
    
    print("\n--- Training Loop ---")
    for epoch in range(5):
        running_loss = 0.0
        for x_feat, x_adj, y_target in train_loader:
            opt.zero_grad()
            
            y_pred = net(x_feat, x_adj)
            batch_loss = loss_fn(y_pred, y_target)
            batch_loss.backward()
            opt.step()
            
            running_loss += batch_loss.item() * x_feat.size(0)
            
        epoch_loss = running_loss / len(train_ds)
        print(f"Iteration [{epoch+1}/5] -> Computed Batch Loss: {epoch_loss:.4f}")
        
    print("\n>>> Local training complete. Weights stable.")

if __name__ == "__main__":
    run_local_training()
