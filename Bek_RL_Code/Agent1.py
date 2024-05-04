# Agent.py

import torch
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import DQN, QTrainer  # Assuming you've defined a DQN model class in model.py
from helper import plot

MAX_MEMORY = 100_000  # Adjust this as needed for your replay memory size
BATCH_SIZE = 1000  # Adjust batch size as needed for training
LR = 0.001  # Learning rate
TARGET_UPDATE_FREQUENCY = 5  # How often to update the target network

class DQNAgent:
    # Initialize DQN Agent
    def __init__(self, input_size, hidden_size, output_size):
        self.n_games = 0
        self.epsilon = 0  # controls the exploration-exploitation tradeoff
        self.gamma = 0.9  # discount rate, same as before
        self.memory = deque(maxlen=MAX_MEMORY)  # replay memory
        self.model = DQN(input_size, hidden_size, output_size)  # policy network
        self.target_model = DQN(input_size, hidden_size, output_size)  # target network
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # set target model to eval mode
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.BATCH_SIZE = BATCH_SIZE
        self.TARGET_UPDATE_FREQUENCY = TARGET_UPDATE_FREQUENCY  # set this to the desired frequency of updating the target network

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        mini_batch = random.sample(self.memory, self.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        
        # Convert to appropriate torch tensor
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones)

        # Predict Q-values with the current state
        q_pred = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Predict next Q-values with the target state
        q_next = self.target_model(next_states).max(1)[0].detach()
        q_next[dones] = 0.0  # Zero out the effects of terminal states

        # Calculate the target Q-values
        q_target = rewards + self.gamma * q_next

        # Compute the loss
        loss = F.mse_loss(q_pred, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
                # Clip gradients to avoid exploding gradient problem
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update target network every TARGET_UPDATE_FREQUENCY games
        if self.n_games % self.TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_short_memory(self, state, action, reward, next_state, done):
        # Convert to torch tensors
        state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        next_state_t = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.long).unsqueeze(0).unsqueeze(0)  # Note the double unsqueeze for a 2D shape
        reward_t = torch.tensor(reward, dtype=torch.float).unsqueeze(0)
        done_t = torch.tensor(done, dtype=torch.bool).unsqueeze(0)
    
        # Predict Q-values with the current state
        q_pred = self.model(state_t)

        # We select the Q-value for the action taken. This is where gather is used
        q_pred = q_pred.gather(1, action_t).squeeze(-1)

        # Predict next Q-values with the next state using the target network
        q_next = self.target_model(next_state_t).max(1)[0].detach()
        q_next[done_t] = 0.0  # Zero out the effects of terminal states

        # Calculate the target Q-values
        q_target = reward_t + self.gamma * q_next

        # Compute the loss
        loss = F.mse_loss(q_pred, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_action(self, state):
        # Exploration vs Exploitation
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            action = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():
                prediction = self.model(state0)
            action = torch.argmax(prediction).item()
        return action
    

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = DQNAgent(input_size=11, hidden_size=256, output_size=3)
    game = SnakeGameAI()
    while True:
        # Get the current state
        state_old = agent.get_state(game)

        # Get action
        action = agent.get_action(state_old)

        # Convert the chosen action to the required format
        final_move = [0, 0, 0]
        final_move[action] = 1

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # Remember
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            # Train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # Update the target network every few games
            if agent.n_games % TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_model()

if __name__ == '__main__':
    train()




DQNAgent.model.save('desired_file_name.pth')  # You can specify a file name or use the default

