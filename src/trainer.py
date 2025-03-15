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
from env import get_dimensions, image_name, img_name, render_state, apply_action, is_terminal
from mcts import MCTS, convert_state_to_tensor,  MCTS_ITERATIONS, compute_intermediate_reward, TIME_PER_MOVE, action_to_index, build_distribution_tensor
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
MAX_STEPS_PER_EPOCH = 100

# OVERRIDE MCTS_ITERATIONS from mcts, for lazy
# LAZY
MCTS_ITERATIONS = 5
 
def load_models(policy_model, value_model, save_path, epoch=None):
    """
    Load saved models from checkpoint.
    If epoch is None, loads the best model according to best_model.txt
    """
    if epoch is None:
        # Try to load best model
        best_model_path = f"{save_path}/best_model.txt"
        if os.path.exists(best_model_path):
            with open(best_model_path, 'r') as f:
                best_info = f.read().strip().split('\n')
                best_epoch = int(best_info[0].split(':')[1].strip())
                best_reward = float(best_info[1].split(':')[1].strip())
                print(f"Loading best model from epoch {best_epoch} with reward {best_reward}")
                epoch = best_epoch
        else:
            print("No best model found. Starting with fresh models.")
            return False
    
    # Load the specified model
    policy_path = f"{save_path}/policy_epoch_{epoch}.pth"
    value_path = f"{save_path}/value_epoch_{epoch}.pth"
    
    if os.path.exists(policy_path) and os.path.exists(value_path):
        policy_model.load_state_dict(torch.load(policy_path))
        value_model.load_state_dict(torch.load(value_path))
        print(f"Loaded models from epoch {epoch}")
        print(f"  - Policy: {policy_path}")
        print(f"  - Value: {value_path}")
        return True
    else:
        print(f"Could not find model files for epoch {epoch}")
        return False

#TODO: mcts_target_policy
def mcts_target_policy(state, policy_model, value_model, iterations = MCTS_ITERATIONS):
    """
    Placeholder function to generate a target policy from MCTS visit counts.
    Replace this with your actual target policy logic.
    """
    # Baseline - uniform distribution
    # Baseline, no target policy
    # num_actions = action_dim  # Adjust if your action space is larger.
    # return torch.ones((1, num_actions), dtype=torch.float32) / num_actions
    """
    Use MCTS to get a distribution (visit counts) over all actions,
    then normalize to get a policy distribution (1 x num_actions).
    """
    best_action, action_distribution = MCTS(
        root_state=state,
        policy_model=policy_model,
        value_model=value_model,
        iterations=iterations,
        render=False
    )
    
    # Convert {action: prob} to a (1 x num_actions) tensor
    normalized_action_distribution = build_distribution_tensor(action_distribution, state)
    return normalized_action_distribution


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

            
# Define the compute_loss function as a standalone function
def compute_loss(state, action, reward, next_state, visual_features, next_visual_features, policy_model, value_model, gamma):
    # Convert state to tensor
    state_tensor = convert_state_to_tensor(state)
    next_state_tensor = convert_state_to_tensor(next_state)
    
    """
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
    """
    
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
    policy_loss = -torch.log(action_probs[0, action_idx])
    
    # Combine the losses
    loss = value_loss + policy_loss
    
    return loss

def compute_loss_with_mcts(state, mcts_dist, reward, next_state, visual_features, next_visual_features, 
                            policy_model, value_model, gamma):
                """
                Example: we combine an MSE value loss with a cross-entropy policy loss
                to match the MCTS distribution.
                """
                # Convert state to tensor
                state_tensor = convert_state_to_tensor(state)
                next_state_tensor = convert_state_to_tensor(next_state)
    
                current_visual_tensor = process_visual_features(visual_features)
                next_visual_tensor = process_visual_features(next_visual_features)
                
                # Compute Value Loss
                predicted_value = value_model(state_tensor, current_visual_tensor)
                next_predicted_value = value_model(next_state_tensor, next_visual_tensor)
                
                #Target = R + gamma * V(next_state)
                target_value = reward + gamma * next_predicted_value.detach()
                
                # MSE loss for value
                value_loss = F.mse_loss(predicted_value, target_value)
                
                #Get policy loss
                predicted_probs = policy_model(state_tensor)

                # We want to match `predicted_probs` to `mcts_dist`. 
                # Let's do a cross-entropy: -\sum(MCTS_dist * log(pred_probs))
                # (1 x action_dim) each
                # We add a small epsilon for numerical stability.
                eps = 1e-8
                policy_loss = -(mcts_dist * torch.log(predicted_probs + eps)).sum()
                
                # 3) combined loss
                loss = value_loss + policy_loss
                
                return loss
            
class Trainer:
    def __init__(self, env, policy_model, value_model, smooth, optimizer, save_path, render_on):
        self.env = env
        self.policy_model = policy_model
        self.value_model = value_model
        self.smooth = smooth
        self.optimizer = optimizer
        self.save_path = save_path
        self.render_on = render_on
        self.losses = []
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Add tracking for best model
        self.best_reward = float('-inf')
        self.best_epoch = -1
        
        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Create progress directory for screenshots
        self.progress_dir = "progress"
        # if smoothing is on, save to a diff directory
        if (self.smooth == True):
            self.progress_dir = "progress_smooth"
            
        os.makedirs(self.progress_dir, exist_ok=True)
        
        # Try to load the best model if it exists
        loaded = load_models(self.policy_model, self.value_model, self.save_path)
        if loaded:
            print("Successfully loaded previous best model.")
        else:
            print("Starting with fresh models.")

    def train(self, num_epochs, max_steps):
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
            while not done and num_moves < max_steps:
                num_moves += 1
                
                # Print epoch and step on the same line
                print(f"[EPOCH {epoch+1}/{num_epochs}] [STEP {num_moves}/{max_steps}]", end=" ")
                
                # Extract visual features for the current state.
                current_visual_features = extract_visual_features(state, image_name)
                
                # Baseline:
                if (self.smooth == False):
                    # Select an action using MCTS
                    action = self.select_action(state)
                    
                # 1) Call MCTS to get both the best action and the distribution
                # Smoothing on
                if (self.smooth):
                    mcts_policy = mcts_target_policy(state, self.policy_model, self.value_model, iterations=MCTS_ITERATIONS)
                    # If you also want the best action from the same MCTS run:
                    # single call:
                    # best_action = pick an action from action_distribution (like argmax)

                    # multiple calls
                    action, action_distribution = MCTS(state, self.policy_model, self.value_model, iterations=MCTS_ITERATIONS)

                # Apply the action: get next_state, reward, etc.
                next_state, reward, done, info = self.step(state, action)
                
                # Extract visual features for the next state.
                next_visual_features = extract_visual_features(next_state, image_name)
                
                # Compute loss using both current and next visual features.
                loss = self.optimize(state, action, reward, next_state,
                                        current_visual_features, next_visual_features)
                
                # Print action, reward and loss information on same line
                print(f"[Action: {action}] [Reward: {reward:.4f}] [Loss: {loss.item():.4f}]")
                
                # Save screenshot every 5 steps
                if self.render_on and num_moves % 50 == 0:
                    # print_puzzle_completion(state) undo after fix
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
        print(f"\n{'='*50}\n[TRAINING COMPLETED]\n{'='*50}")
        print(f"Best model was from epoch {self.best_epoch} with reward {self.best_reward:.4f}")
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
        if (self.smooth == False):
            """Compute loss, perform backpropagation, and update the networks."""
            loss = compute_loss(state, action, reward, next_state, visual_features, next_visual_features, 
                                self.policy_model, self.value_model, GAMMA)
        if (self.smooth == True):
            # 1) Get MCTS distribution for current state
            mcts_dist = mcts_target_policy(state, self.policy_model, self.value_model, iterations=MCTS_ITERATIONS)
            loss = compute_loss_with_mcts(state, mcts_dist, reward, next_state, visual_features, next_visual_features, 
                                self.policy_model, self.value_model, GAMMA)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        return loss

    def save_model(self, epoch):
        """Save the current policy and value networks."""
        torch.save(self.policy_model.state_dict(), f"{self.save_path}/policy_epoch_{epoch}.pth")
        torch.save(self.value_model.state_dict(), f"{self.save_path}/value_epoch_{epoch}.pth")
        print(f"  [Saved model for epoch {epoch}]")

    def save_best_model(self, epoch, reward):
        """Save information about the best model."""
        with open(f"{self.save_path}/best_model.txt", 'w') as f:
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Reward: {reward}\n")
            f.write(f"Episode Length: {self.episode_lengths[-1]}\n")
            f.write(f"Loss: {self.losses[-1]}\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"  [Updated best_model.txt with information about epoch {epoch}]")

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

def calculate_puzzle_completion(state, centroids_file="Datasets/puzzle_centroids.json"):
    """
    Calculate how close the puzzle pieces are to their target positions.
    
    Args:
        state: The current puzzle state containing piece positions; 
               `state.assembly` is assumed to be a 2D array or similar structure.
        centroids_file: Path to the JSON file with target centroids.
        
    Returns:
        completion_percentage (float): Overall percentage of puzzle completion.
        piece_distances (dict): Dictionary of distances for each piece.
        mse_distance (float): Mean squared error of distances.
    """
    # Load centroids data
    with open(centroids_file, 'r') as f:
        centroids_data = json.load(f)
    
    # Extract piece information from state
    piece_positions = {}
    for row_index, row in enumerate(state.assembly):
        for col_index, piece_id in enumerate(row):
            if piece_id is not None:
                piece_positions[piece_id] = {
                    "x": col_index,
                    "y": row_index
                }

    # Calculate distance for each piece
    max_distance = 0
    total_distance = 0
    piece_distances = {}
    
    for piece_info in centroids_data["pieces"]:
        piece_id = piece_info["id"]
        target_x = piece_info["centroid"]["x"]
        target_y = piece_info["centroid"]["y"]
        
        if piece_id in piece_positions:
            current_x = piece_positions[piece_id]["x"]
            current_y = piece_positions[piece_id]["y"]
            
            # Calculate Euclidean distance
            distance = math.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)
            piece_distances[piece_id] = distance
            total_distance += distance
            
            # Track theoretical maximum distance (diagonal of screen)
            # This assumes a 974x758 screen based on the screenshot dimensions
            # TODO: bro how did u get this number lmao
            max_distance += math.sqrt(974**2 + 758**2)
        else:
            # Missing piece in the current assembly
            piece_distances[piece_id] = float('inf')
            total_distance += math.sqrt(974**2 + 758**2)  # Add max possible distance for missing pieces
            max_distance += math.sqrt(974**2 + 758**2)
    
    # Calculate completion percentage (inverted - closer means higher percentage)
    # Using an exponential decay formula to make percentage more meaningful
    if max_distance == 0:  # Safety check
        completion_percentage = 100.0
    else:
        normalized_distance = total_distance / max_distance
        completion_percentage = 100 * math.exp(-5 * normalized_distance)  # Exponential scaling
        
    # ------------------------------------------------------------
    # 5. Calculate MSE (Mean Squared Error) of distances
    # ------------------------------------------------------------
    piece_count = len(piece_distances)
    if piece_count == 0:
        mse_distance = 0.0
    else:
        sum_of_squares = 0.0
        for dist in piece_distances.values():
            if dist == float('inf'):
                # If a piece is missing, treat that distance as max possible
                sum_of_squares += max_distance**2
            else:
                sum_of_squares += dist**2
        mse_distance = sum_of_squares / piece_count
    
    return completion_percentage, piece_distances, mse_distance

def print_puzzle_completion(state, centroids_file="Datasets/puzzle_centroids.json"):
    """
    Print information about puzzle completion status.
    """
    completion_percentage, piece_distances, mse_distance = calculate_puzzle_completion(state, centroids_file)
    
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
    #pygame.init()
    #pygame.display.set_mode((1, 1))  # or minimal, (1,1)
    # use render_state from env
    # if render off, pygame.display.set_mode((1, 1)) for minimal display
    # Instantiate the Trainer using the environment, models, and optimizer.
    
    #trainer = Trainer(env, policy_model, value_model, False, optimizer, save_path="checkpoints", render_on = True, max_steps = 100)
    #trainer.train(num_epochs=100, max_steps = 100)
    
    # smooth, and lazy
    trainer = Trainer(env, policy_model, value_model, True, optimizer, save_path="checkpoints_smooth", render_on = True)
    trainer.train(num_epochs=5, max_steps=5)
    