import pygame
import torch
import env
from env import SCREEN_WIDTH, SCREEN_HEIGHT
from trainer import Trainer  # If you keep your Trainer in a separate file
#from env import MyEnvironment
from models import PolicyNetwork, ValueNetwork

def main():
    # Decide whether you want to render during training
    render_on = True  # or True if you want a full window

    if render_on:
        pygame.init()
        pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))  # Full-size window for debugging
    else:
        # Still need a display for image loading, so minimal:
        pygame.init()
        pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    # Create your environment
    #env = MyEnvironment()
    # Create your models
    policy_model = PolicyNetwork(state_dim=30, action_dim=200)
    value_model = ValueNetwork(state_dim=30, visual_dim=128)
    # Create an optimizer
    optimizer = torch.optim.Adam(list(policy_model.parameters()) + list(value_model.parameters()), lr=1e-3)

    # Instantiate the Trainer
    trainer = Trainer(
        env=env,
        policy_model=policy_model,
        value_model=value_model,
        optimizer=optimizer,
        save_path="checkpoints",
        render_on=render_on  # pass the flag
    )
    trainer.train(num_epochs=100)

    # Quit pygame when done
    pygame.quit()

if __name__ == "__main__":
    main()
