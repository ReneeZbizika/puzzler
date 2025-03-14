import os
import torch
import torch.optim as optim
import numpy
import matplotlib as plt
import pygame

import env
from env import get_dimensions, image_name, img_name, render_state
from mcts import MCTS, convert_state_to_tensor,  MCTS_ITERATIONS
#state_dim, action_dim, visual_dim,
from models import PolicyNetwork, ValueNetwork

from utils_features import extract_visual_features

# Get dimensions dynamically from env.py
state_dim, action_dim, visual_dim = get_dimensions()

# Create the models (initially untrained)
policy_model = PolicyNetwork(state_dim, action_dim)
value_model = ValueNetwork(state_dim, visual_dim)

# Create an optimizer for both networks (here we use Adam)
optimizer = optim.Adam(list(policy_model.parameters()) + list(value_model.parameters()), lr=1e-3)

def load_models(policy_model, value_model, save_path, epoch):
    """Load saved models from checkpoint."""
    policy_model.load_state_dict(torch.load(f"{save_path}/policy_epoch_{epoch}.pth"))
    value_model.load_state_dict(torch.load(f"{save_path}/value_epoch_{epoch}.pth"))
    print(f"Loaded models from {save_path}/policy_epoch_{epoch}.pth and value_epoch_{epoch}.pth")
    
#load_models(policy_model, value_model, "checkpoints", epoch=50)

#TODO: mcts_target_policy
def mcts_target_policy(state):
    """
    Placeholder function to generate a target policy from MCTS visit counts.
    Replace this with your actual target policy logic.
    """
    num_actions = action_dim  # Adjust if your action space is larger.
    return torch.ones((1, num_actions), dtype=torch.float32) / num_actions


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
    def __init__(self, env, policy_model, value_model, optimizer, save_path, render_on):
        self.env = env
        self.policy_model = policy_model
        self.value_model = value_model
        self.optimizer = optimizer
        self.save_path = save_path
        self.render_on = render_on
        self.losses = []
        self.episode_rewards = []
        self.episode_lengths = []

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            state = self.env.reset()  # Reset returns a State object
            total_reward = 0
            num_moves = 0
            done = False
            while not done:
                # Extract visual features for the current state.
                current_visual_features = extract_visual_features(state, image_name)
                
                # Select an action using MCTS (or any other method).
                action = self.select_action(state)
                
                # Apply the action: get next_state, reward, etc.
                next_state, reward, done, info = self.env.step(action)
                
                # Extract visual features for the next state.
                next_visual_features = extract_visual_features(next_state, image_name)
                
                # Compute loss using both current and next visual features.
                loss = self.optimize(state, action, reward, next_state,
                                    current_visual_features, next_visual_features)
                
                state = next_state
                total_reward += reward
                num_moves += 1
                
            self.losses.append(loss.item())
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(num_moves)
            self.save_model(epoch)
            self.log_metrics(epoch)
        self.plot_progress()

    def plot_progress(self):
        """Plot loss, total reward, and moves per episode over training epochs."""
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
        """Select the next action using MCTS."""
        action = MCTS(state, self.policy_model, self.value_model, iterations=MCTS_ITERATIONS)
        return action

    def optimize(self, state, action, reward, next_state, visual_features, next_visual_features):
        """Compute loss, perform backpropagation, and update the networks."""
        loss = compute_loss(state, action, reward, next_state, visual_features, next_visual_features)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def save_model(self, epoch):
        """Save the current policy and value networks."""
        torch.save(self.policy_model.state_dict(), f"{self.save_path}/policy_epoch_{epoch}.pth")
        torch.save(self.value_model.state_dict(), f"{self.save_path}/value_epoch_{epoch}.pth")

    def log_metrics(self, epoch):
        """Log training metrics."""
        print(f"Epoch {epoch}: Model saved and metrics logged.")

if __name__ == "__main__":
    pygame.init()
    pygame.display.set_mode((1, 1))  # or minimal, (1,1)
    # use render_state from env
    # if render off, pygame.display.set_mode((1, 1)) for minimal display
    # Instantiate the Trainer using the environment, models, and optimizer.
    trainer = Trainer(env, policy_model, value_model, optimizer, save_path="checkpoints", render_on = True)
    trainer.train(num_epochs=100)