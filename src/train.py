import torch
import os
from trainer import Trainer
from models import PolicyNetwork, ValueNetwork
from env import initialize_state
import torch.optim as optim

# Set dimensions (adjust based on your state representation)
from env import get_dimensions

# Dynamically retrieve dimensions
STATE_DIM, ACTION_DIM, VISUAL_DIM = get_dimensions()

# Initialize models
policy_model = PolicyNetwork(STATE_DIM, ACTION_DIM)
value_model = ValueNetwork(STATE_DIM, VISUAL_DIM)

# Optimizer
optimizer = optim.Adam(list(policy_model.parameters()) + list(value_model.parameters()), lr=1e-3)

# Create environment
env = initialize_state()  # Make sure env.py correctly initializes the state

# Define model save path
SAVE_PATH = "checkpoints"

# Ensure checkpoint directory exists
os.makedirs(SAVE_PATH, exist_ok=True)

# Initialize Trainer
trainer = Trainer(env, policy_model, value_model, optimizer, save_path=SAVE_PATH)

from mcts import MCTS

# Load trained models
policy_model.load_state_dict(torch.load("checkpoints/policy_epoch_100.pth"))
value_model.load_state_dict(torch.load("checkpoints/value_epoch_100.pth"))

# Initialize environment
state = env.reset()

# Run MCTS to pick an action using the trained models
# action = MCTS(state, iterations=100)
# print(f"Selected action: {action}")

# Run training for 100 epochs
if __name__ == "__main__":
    trainer.train(num_epochs=100)
    

