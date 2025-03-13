import os
import torch
import torch.optim as optim
import numpy
import matplotlib as plt

from mcts import convert_state_to_tensor

from models import PolicyNetwork, ValueNetwork

#TODO
# Ensure the model learns from both numerical and visual information during training.

#TODO
# # Define dimensions (you need to set these based on your environment/state representation)
state_dim = 100    # e.g., flattened state vector length
action_dim = 10    # e.g., number of possible actions

# Create the models (initially untrained)
policy_model = PolicyNetwork(state_dim, action_dim)
value_model = ValueNetwork(state_dim)

# Create an optimizer for both networks (here we use Adam)
optimizer = optim.Adam(list(policy_model.parameters()) + list(value_model.parameters()), lr=1e-3)

def load_models(policy_model, value_model, save_path, epoch):
    """Load saved models from checkpoint."""
    policy_model.load_state_dict(torch.load(f"{save_path}/policy_epoch_{epoch}.pth"))
    value_model.load_state_dict(torch.load(f"{save_path}/value_epoch_{epoch}.pth"))
    print(f"Loaded models from {save_path}/policy_epoch_{epoch}.pth and value_epoch_{epoch}.pth")
    
#load_models(policy_model, value_model, "checkpoints", epoch=50)


# This is a conceptual example. 
# In practice, we might have more sophisticated target generation 
# (for example, using the visit count distribution from MCTS as the target for the policy network).

#TODO
def compute_loss(state, action, reward, next_state, visual_features, next_visual_features, gamma=0.99):
    # Convert states to tensors
    state_tensor = convert_state_to_tensor(state)
    next_state_tensor = convert_state_to_tensor(next_state)
    
    visual_tensor = torch.tensor(visual_features, dtype=torch.float32).unsqueeze(0)
    next_visual_tensor = torch.tensor(next_visual_features, dtype=torch.float32).unsqueeze(0)

    # Forward pass through policy and value networks
    predicted_policy = policy_model(state_tensor)
    predicted_value = value_model(state_tensor, visual_tensor)

    # Assume target policy from MCTS visit counts
    target_policy = mcts_target_policy(state)

    policy_loss = -torch.sum(target_policy * torch.log(predicted_policy + 1e-8))

    # Compute target value with bootstrapping
    with torch.no_grad():
        target_value = reward + gamma * value_model(next_state_tensor, next_visual_tensor)

    value_loss = (predicted_value - target_value).pow(2).mean()

    return policy_loss + value_loss

#TODO
class Trainer:
    def __init__(self, env, policy_model, value_model, optimizer, save_path):
        self.env = env
        self.policy_model = policy_model
        self.value_model = value_model
        self.optimizer = optimizer
        self.save_path = save_path
        self.losses = []
        self.episode_rewards = []
        self.episode_lengths = []

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            state = self.env.reset()  # Initialize environment with a new image, if applicable
            #TODO 
            total_reward = 0
            num_moves = 0
            done = False
            while not done:
                action = self.select_action(state)  # e.g., using MCTS
                # action, target_policy = MCTS(state, iterations=100)  # Modify MCTS to return target_policy if needed
        
                next_state, reward, done, info = self.env.step(action)
                loss = self.optimize(state, action, reward, next_state)
                
                state = next_state
            
            # Save progress
            self.losses.append(loss.item())
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(num_moves)
                
            self.save_model(epoch)
            self.log_metrics(epoch)
        self.plot_progress()
        
    def plot_progress(self):
        """Plots loss, rewards, and episode length over epochs."""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(self.losses, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(self.episode_rewards, label="Total Reward")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(self.episode_lengths, label="Moves per Episode")
        plt.xlabel("Epoch")
        plt.ylabel("Moves")
        plt.legend()

        plt.show()
        
    def select_action(self, state):
        # Run MCTS to decide on the best action given the current state
        action = MCTS(state, iterations=100)  # Placeholder for your MCTS call
        return action

    def optimize(self, state, action, reward, next_state):
        # Compute loss and update your models with PyTorch
        # For example, compute policy loss and value loss and backpropagate
        loss = compute_loss(state, action, reward, next_state)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, epoch):
        # Save the policy and value networks
        torch.save(self.policy_model.state_dict(), f"{self.save_path}/policy_epoch_{epoch}.pth")
        torch.save(self.value_model.state_dict(), f"{self.save_path}/value_epoch_{epoch}.pth")

    def log_metrics(self, epoch):
        # Log your training metrics, e.g., loss, reward, etc.
        print(f"Epoch {epoch}: Model saved and metrics logged.")

# Usage
if __name__ == "__main__":
    trainer = Trainer(env, policy_model, value_model, optimizer, save_path="checkpoints")
    trainer.train(num_epochs=100)
