import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiplexGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_layers=3, slope=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.slope = slope
        
        # Linear layer weights for mapping node attributes
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        # Attention parameter vector
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        
        # Standard initializations for stability
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        batch_size = h.size(0)
        nodes = h.size(1)
        
        # Project our node spectral features
        Wh = torch.matmul(h, self.W) # Out: (batch, nodes, out_features)
        
        # Set up combinations for the attention broadcast
        Wh_1 = Wh.repeat_interleave(nodes, dim=1)
        Wh_2 = Wh.repeat(1, nodes, 1)
        pairs = torch.cat([Wh_1, Wh_2], dim=-1).view(batch_size, nodes, nodes, -1)
        
        # Calculate raw scores across paths
        e_scores = torch.matmul(pairs, self.a).squeeze(-1)
        e_scores = F.leaky_relu(e_scores, negative_slope=self.slope)
        
        # Broadcast scores across our 3 target cross-frequency graph layers
        e_tensor = e_scores.unsqueeze(1).repeat(1, self.num_layers, 1, 1)
        
        # Mask out connection elements that fall below our 0.45 noise threshold
        mask_val = -1e9 * torch.ones_like(e_tensor)
        attn_weights = torch.where(adj > 0, e_tensor, mask_val)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Process message aggregation independently for each frequency band layer
        aggregated = []
        for idx in range(self.num_layers):
            layer_attn = attn_weights[:, idx, :, :]
            layer_out = torch.matmul(layer_attn, Wh)
            aggregated.append(layer_out)
            
        return F.elu(torch.stack(aggregated, dim=1))

class MultiplexBrainEngine(nn.Module):
    def __init__(self, channels=32, features=5, layers=3, hidden=32):
        super().__init__()
        self.gat = MultiplexGATLayer(in_features=features, out_features=hidden, num_layers=layers)
        # Final classification network layer mapping directly to trajectory target score
        self.classifier = nn.Linear(layers * channels * hidden, 1)
        
    def forward(self, x, adj):
        # Extract cross-frequency spatial feature maps
        embeddings = self.gat(x, adj)
        # Flatten securely for binary output mapping step
        flat_features = embeddings.view(embeddings.size(0), -1)
        return torch.sigmoid(self.classifier(flat_features))

if __name__ == "__main__":
    # Local dry-run verification check
    model = MultiplexBrainEngine()
    x_test = torch.randn(1, 32, 5)
    adj_test = torch.rand(1, 3, 32, 32)
    output = model(x_test, adj_test)
    print(">>> Model local trace test completed. Output size:", output.shape)
