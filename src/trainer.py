import os
import torch
import torch.optim as optim
import numpy
import matplotlib.pyplot as plt
import pygame
import torch.nn.functional as F
import datetime
import json
import math

import env
from env import get_dimensions, image_name, render_state, apply_action, is_terminal, centroids_folder
from mcts import MCTS, convert_state_to_tensor,  MCTS_ITERATIONS, compute_intermediate_reward, TIME_PER_MOVE, action_to_index
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

# Add this constant at the top of your file, outside any class
GAMMA = 0.99  # Standard discount factor for reinforcement learning

# Add a constant for max steps per epoch
MAX_STEPS_PER_EPOCH = 5

def load_models(policy_model, value_model, save_path):
    # Import the flag from env
    from env import ENABLE_MODEL_LOADING
    
    # Skip loading if disabled
    if not ENABLE_MODEL_LOADING:
        print("Model loading disabled. Starting with fresh models.")
        return False
        
    # Continue with existing loading logic
    policy_path = os.path.join(save_path, "policy_model.pth")
    value_path = os.path.join(save_path, "value_model.pth")
    
    if not os.path.exists(policy_path) or not os.path.exists(value_path):
        print("No saved models found. Starting with fresh models.")
        return False
    
    try:
        policy_model.load_state_dict(torch.load(policy_path))
        value_model.load_state_dict(torch.load(value_path))
        print("Models loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Starting with fresh models.")
        return False

#TODO: mcts_target_policy
def mcts_target_policy(state):
    """
    Placeholder function to generate a target policy from MCTS visit counts.
    Replace this with your actual target policy logic.
    """
    num_actions = action_dim  # Adjust if your action space is larger.
    return torch.ones((1, num_actions), dtype=torch.float32) / num_actions

# Define the compute_loss function as a standalone function
def compute_loss(state, action, reward, next_state, visual_features, next_visual_features, policy_model, value_model, gamma):
    # Convert state to tensor
    state_tensor = convert_state_to_tensor(state)
    next_state_tensor = convert_state_to_tensor(next_state)
    
    # Process visual features to ensure they have the correct shape (1, 2)
    def process_visual_features(features):
        if isinstance(features, (float, numpy.float64, numpy.float32)):
            # If it's a single value, create a list with two elements
            features = [float(features), 0.0]
        elif hasattr(features, '__iter__'):
            # If it's already an iterable, convert to list
            features = list(features)
            # Make sure we have exactly 2 features
            if len(features) < 2:
                features = features + [0.0] * (2 - len(features))
            elif len(features) > 2:
                features = features[:2]
        else:
            # Fallback for any other type
            features = [0.0, 0.0]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    # Process both current and next visual features
    visual_tensor = process_visual_features(visual_features)
    next_visual_tensor = process_visual_features(next_visual_features)
    
    # Ensure tensors have the correct shape
    assert visual_tensor.shape == (1, 2), f"Visual tensor has incorrect shape: {visual_tensor.shape}, expected (1, 2)"
    assert next_visual_tensor.shape == (1, 2), f"Next visual tensor has incorrect shape: {next_visual_tensor.shape}, expected (1, 2)"
    
    # Get the predicted value for the current state
    predicted_value = value_model(state_tensor, visual_tensor)
    
    # Get the predicted value for the next state
    next_predicted_value = value_model(next_state_tensor, next_visual_tensor)
    
    # Calculate the target value using the reward and the discounted next state value
    target_value = reward + gamma * next_predicted_value
    
    # Calculate the value loss (MSE)
    value_loss = F.mse_loss(predicted_value, target_value.detach())
    
    # Get the action probabilities from the policy network
    action_probs = policy_model(state_tensor)
    
    # Convert the action to an index
    action_idx = action_to_index(state, action)
    
    # Calculate the policy loss (negative log likelihood of the taken action)
    # alphazzle says to use categorical cross-entropy
    policy_loss = -torch.log(action_probs[0, action_idx])
    
    # Combine the losses
    loss = value_loss + policy_loss
    
    return loss

#TODO
class Trainer:
    def __init__(self, env, policy_model, value_model, optimizer, save_path, render_on):
        self.env = env
        self.policy_model = policy_model
        self.value_model = value_model
        self.optimizer = optimizer
        
        # Create image-specific save path
        self.base_save_path = save_path
        self.image_name = image_name  # From env import
        self.save_path = os.path.join(save_path, f"{self.image_name}")
        
        self.render_on = render_on
        self.losses = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.completion_rates = []
        self.avg_rewards_per_step = []
        
        # Add tracking for best model
        self.best_reward = float('-inf')
        self.best_epoch = -1
        
        # Create image-specific save directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
        
        # Create progress directory for screenshots (also image-specific)
        self.progress_dir = os.path.join("progress", f"{self.image_name}")
        os.makedirs(self.progress_dir, exist_ok=True)
        
        # Try to load the best model for this specific image if it exists
        loaded = self.load_models()
        if loaded:
            print(f"Successfully loaded previous best model for {self.image_name}.")
        else:
            print(f"Starting with fresh models for {self.image_name}.")

    def train(self, num_epochs):
        print(f"\n{'='*50}\n[STARTING TRAINING: {num_epochs} EPOCHS]\n{'='*50}")
        for epoch in range(num_epochs):
            # Create epoch directory for screenshots
            epoch_dir = os.path.join(self.progress_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            print(f"Created screenshot directory: {epoch_dir}")
            
            self.env = env  # Use the existing environment
            state = self.env.reset()  # Reset returns a State object
            
            # Add this line to print the separator after the pieces are loaded
            print("="*50)
            
            total_reward = 0
            num_moves = 0
            done = False
            
            # Save initial state screenshot
            if self.render_on:
                self.save_screenshot(epoch_dir, num_moves, state)
                print(f"[Saved initial puzzle state screenshot]")
            
            # Add a step limit to prevent infinite loops
            while not done and num_moves < MAX_STEPS_PER_EPOCH:
                num_moves += 1
                
                # Print epoch and step on the same line
                print(f"[EPOCH {epoch+1}/{num_epochs}] [STEP {num_moves}/{MAX_STEPS_PER_EPOCH}]", end=" ")
                
                # Extract visual features for the current state.
                current_visual_features = extract_visual_features(state, image_name)
                
                # Select an action using MCTS (or any other method).
                action = self.select_action(state)
                
                # Apply the action: get next_state, reward, etc.
                next_state, reward, done, info = self.step(state, action)
                
                # Extract visual features for the next state.
                next_visual_features = extract_visual_features(next_state, image_name)
                
                # Compute loss using both current and next visual features.
                loss = self.optimize(state, action, reward, next_state,
                                    current_visual_features, next_visual_features)
                
                # Print action, reward and loss information on same line
                print(f"[Action: {action}] [Reward: {reward:.4f}] [Loss: {loss.item():.4f}]")
                
                print_puzzle_completion(state)

                if self.render_on and num_moves % 50 == 0:
                    self.save_screenshot(epoch_dir, num_moves, state)
                    print(f"[Saved puzzle state screenshot for step {num_moves}]")
                
                state = next_state
                total_reward += reward
            
            # Save final state screenshot
            if self.render_on:
                self.save_screenshot(epoch_dir, num_moves, state, is_final=True)
                print(f"[Saved final puzzle state screenshot]")
            
            self.losses.append(loss.item())
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(num_moves)
            
            # Save the model for this epoch
            self.save_model(epoch)
            
            # Check if this is the best model so far
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.best_epoch = epoch
                self.save_best_model(epoch, total_reward)
                print(f"  [NEW BEST MODEL at epoch {epoch} with reward {total_reward:.4f}]")
            
            self.log_metrics(epoch)
            
            # At the end of each episode, after the loop:
            # Capture the final completion rate for this episode
            final_completion_rate = print_puzzle_completion(state)
            self.completion_rates.append(final_completion_rate)
            
            # Calculate average reward per step
            avg_reward = total_reward / num_moves if num_moves > 0 else 0
            self.avg_rewards_per_step.append(avg_reward)
            
        print(f"\n{'='*50}\n[TRAINING COMPLETED]\n{'='*50}")
        print(f"Best model was from epoch {self.best_epoch} with reward {self.best_reward:.4f}")
        self.plot_progress()

    def plot_progress(self):
        """Plot training metrics including puzzle completion rate and average reward per step."""
        # Create graphs directory if it doesn't exist
        graphs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "data", "evaluation", "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        
        # Create timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with a 3x2 grid of subplots
        plt.figure(figsize=(15, 12))
        
        plt.subplot(3, 2, 1)
        plt.plot(self.losses, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        
        plt.subplot(3, 2, 2)
        plt.plot(self.episode_rewards, label="Total Reward")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.title("Episode Total Reward")
        plt.legend()
        
        plt.subplot(3, 2, 3)
        plt.plot(self.episode_lengths, label="Steps")
        plt.xlabel("Epoch")
        plt.ylabel("Steps")
        plt.title("Episode Length")
        plt.legend()
        
        plt.subplot(3, 2, 4)
        plt.plot(self.completion_rates, label="Completion %", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Completion %")
        plt.title("Puzzle Completion Rate")
        plt.legend()
        
        plt.subplot(3, 2, 5)
        plt.plot(self.avg_rewards_per_step, label="Avg Reward/Step", color="purple", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.title("Average Reward Per Step")
        plt.legend()
        
        plt.tight_layout()
        
        # Save the combined figure
        combined_path = os.path.join(graphs_dir, f"training_metrics_{timestamp}.png")
        plt.savefig(combined_path)
        print(f"Saved training metrics to {combined_path}")
        
        # Also save individual plots
        
        # Average reward per step plot
        plt.figure(figsize=(8, 6))
        plt.plot(self.avg_rewards_per_step, label="Avg Reward/Step", color="purple", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.title("Average Reward Per Step")
        plt.grid(True)
        plt.legend()
        avg_reward_path = os.path.join(graphs_dir, f"avg_reward_per_step_{timestamp}.png")
        plt.savefig(avg_reward_path)
        
        # Completion rate plot
        plt.figure(figsize=(8, 6))
        plt.plot(self.completion_rates, label="Completion %", color="green", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Completion %")
        plt.title("Puzzle Completion Rate")
        plt.grid(True)
        plt.legend()
        completion_path = os.path.join(graphs_dir, f"completion_rate_{timestamp}.png")
        plt.savefig(completion_path)
        
        # Still show the plots if in an interactive environment
        plt.show()

    def select_action(self, state):
        """Select the next action using MCTS."""
        action = MCTS(state, self.policy_model, self.value_model, iterations=MCTS_ITERATIONS)
        return action

    def optimize(self, state, action, reward, next_state, visual_features, next_visual_features):
        """Compute loss, perform backpropagation, and update the networks."""
        loss = compute_loss(state, action, reward, next_state, visual_features, next_visual_features, 
                            self.policy_model, self.value_model, GAMMA)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def save_model(self, epoch):
        """Save the current policy and value networks with image-specific naming."""
        torch.save(self.policy_model.state_dict(), f"{self.save_path}/policy_epoch_{epoch}.pth")
        torch.save(self.value_model.state_dict(), f"{self.save_path}/value_epoch_{epoch}.pth")
        print(f"  [Saved model for {self.image_name} epoch {epoch}]")

    def save_best_model(self, epoch, reward):
        """Save information about the best model with image-specific naming."""
        # Save model weights as the default files for easy loading
        torch.save(self.policy_model.state_dict(), f"{self.save_path}/policy_model.pth")
        torch.save(self.value_model.state_dict(), f"{self.save_path}/value_model.pth")
        
        # Save metadata about this best model
        with open(f"{self.save_path}/best_model.txt", 'w') as f:
            f.write(f"Image: {self.image_name}\n")
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Reward: {reward}\n")
            f.write(f"Episode Length: {self.episode_lengths[-1]}\n")
            f.write(f"Loss: {self.losses[-1]}\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"  [Updated best_model.txt for {self.image_name} at epoch {epoch}]")

    def log_metrics(self, epoch):
        """Log training metrics."""
        print(f"\n{'='*10} [EPOCH {epoch+1} SUMMARY] {'='*10}")
        print(f"  [Total Moves: {self.episode_lengths[-1]}]")
        print(f"  [Total Reward: {self.episode_rewards[-1]:.4f}]")
        print(f"  [Final Loss: {self.losses[-1]:.4f}]")
        print(f"  [Model saved to: {self.save_path}/policy_epoch_{epoch}.pth]")
        print(f"{'='*40}")

    def step(self, state, action):
        """
        Execute an action in the given state and return the next state, reward, done flag, and info.
        """
        # Apply the action to get the next state
        next_state = apply_action(state, action)
        
        # Calculate reward
        reward = compute_intermediate_reward(state, action, TIME_PER_MOVE)
        
        # Check if the episode is done
        done = is_terminal(next_state)
        
        # Additional info
        info = {}
        
        return next_state, reward, done, info

    def save_screenshot(self, epoch_dir, step, state, is_final=False):
        """Save a screenshot of the current pygame display."""
        if pygame.display.get_surface() is None:
            print("Warning: No pygame display available for screenshot")
            return
        
        screen = pygame.display.set_mode((974, 758))

        # Force a render of the current state
        render_state(screen, state)  # Now state is properly passed as a parameter
        
        # Ensure the display is updated
        pygame.display.flip()
        
        # Optional: small delay to ensure rendering is complete
        pygame.time.delay(100)  # 100ms delay
        
        # Get the pygame display surface
        surface = pygame.display.get_surface()
        
        # Create filename
        if is_final:
            filename = f"final_step_{step}.png"
        else:
            filename = f"step_{step}.png"
        
        filepath = os.path.join(epoch_dir, filename)
        
        # Save the screenshot
        pygame.image.save(surface, filepath)
        
        return filepath

    def load_models(self):
        """Load models specific to the current image if they exist."""
        # Skip loading if disabled in env
        try:
            from env import ENABLE_MODEL_LOADING
            if not ENABLE_MODEL_LOADING:
                print("Model loading disabled. Starting with fresh models.")
                return False
        except (ImportError, AttributeError):
            # If the flag doesn't exist, default to attempting to load
            pass
        
        # Look for image-specific model files
        policy_path = os.path.join(self.save_path, "policy_model.pth")
        value_path = os.path.join(self.save_path, "value_model.pth")
        
        if not os.path.exists(policy_path) or not os.path.exists(value_path):
            print(f"No saved models found for {self.image_name}. Starting with fresh models.")
            return False
        
        try:
            print(f"Loading models for {self.image_name} from {self.save_path}")
            self.policy_model.load_state_dict(torch.load(policy_path))
            self.value_model.load_state_dict(torch.load(value_path))
            
            # Load best model metadata if available
            best_model_info_path = os.path.join(self.save_path, "best_model.txt")
            if os.path.exists(best_model_info_path):
                with open(best_model_info_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Reward:"):
                            try:
                                self.best_reward = float(line.split(":")[1].strip())
                            except:
                                pass
                        elif line.startswith("Epoch:"):
                            try:
                                self.best_epoch = int(line.split(":")[1].strip())
                            except:
                                pass
            
            print(f"Best model info loaded: Epoch {self.best_epoch}, Reward {self.best_reward:.4f}")
            
            print("Models loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Starting with fresh models.")
            return False

def calculate_puzzle_completion(state, centroids_file=centroids_folder):
    """
    Calculate how close the puzzle pieces are to their target positions.
    
    Args:
        state: The current puzzle state containing piece positions
        centroids_file: Path to the JSON file with target centroids
        
    Returns:
        completion_percentage: Overall percentage of puzzle completion
        piece_distances: Dictionary of distances for each piece
    """
    # Load centroids data
    with open(centroids_file, 'r') as f:
        centroids_data = json.load(f)
    
    piece_positions = {}
    for piece_id, piece_obj in state.pieces.items():
        formatted_id = f"piece_{piece_id}"
        piece_positions[formatted_id] = {"x": piece_obj.x, "y": piece_obj.y}

    # Calculate distance for each piece
    max_distance = 0
    total_distance = 0
    piece_distances = {}
    
    for piece_info in centroids_data["pieces"]:
        piece_id = piece_info["id"]
        target_x = piece_info["centroid"]["x"]
        target_y = piece_info["centroid"]["y"]
        
        max_single_piece_distance = math.sqrt(974**2 + 758**2)
        
        if piece_id in piece_positions:
            current_x = piece_positions[piece_id]["x"]
            current_y = piece_positions[piece_id]["y"]
            
            # Calculate Euclidean distance
            distance = math.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)
            piece_distances[piece_id] = distance
            total_distance += distance
            
            # Track theoretical maximum distance
            max_distance += max_single_piece_distance
        else:
            piece_distances[piece_id] = float('inf')
            total_distance += max_single_piece_distance
            max_distance += max_single_piece_distance
            print(f"Piece {piece_id} not found in current state")
    
    # Calculate completion percentage
    if max_distance < 1e-6:
        completion_percentage = 100.0
    else:
        normalized_distance = total_distance / max_distance
        completion_percentage = 100 * math.exp(-5 * normalized_distance)
    
    return completion_percentage, piece_distances

def print_puzzle_completion(state, centroids_file=centroids_folder):
    """
    Print information about puzzle completion status.
    """
    completion_percentage, piece_distances = calculate_puzzle_completion(state, centroids_file)
    
    print(f"\n{'='*50}")
    print(f"PUZZLE COMPLETION: {completion_percentage:.2f}%")
    print(f"{'='*50}")
    
    # Sort pieces by distance (closest to furthest)
    sorted_distances = sorted(piece_distances.items(), key=lambda x: x[1])
    
    print("PIECE POSITIONS (closest to target first):")
    for i, (piece_id, distance) in enumerate(sorted_distances):
        print(f"  {i+1:2d}. {piece_id}: {distance:.2f} pixels from target position")
    print(f"{'='*50}\n")
    
    return completion_percentage

if __name__ == "__main__":
    # Instantiate the Trainer using the environment, models, and optimizer.
    trainer = Trainer(env, policy_model, value_model, optimizer, save_path="checkpoints", render_on=True)
    trainer.train(num_epochs=5)