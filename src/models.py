# models.py
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# Modify the network to accept both state information and visual features as input.
import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, visual_dim):
        super(ValueNetwork, self).__init__()
        
        # Numerical features (state representation)
        self.state_fc = nn.Linear(state_dim, 128)

        # Visual features (processed edges, similarity, etc.)
        self.visual_fc = nn.Linear(visual_dim, 128)

        # Combine both
        self.combined_fc = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, state_input, visual_input):
        state_embedding = F.relu(self.state_fc(state_input))
        visual_embedding = F.relu(self.visual_fc(visual_input))

        combined = torch.cat([state_embedding, visual_embedding], dim=1)
        combined = F.relu(self.combined_fc(combined))

        return self.output(combined)

    



