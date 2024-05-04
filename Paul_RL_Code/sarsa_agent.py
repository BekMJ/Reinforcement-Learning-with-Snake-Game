import numpy as np
import random
from collections import deque
from game import SnakeGameAI, Direction, Point
from plotter import plot

class SARSAAgent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 1 #start with full exploratio
        self.epsilon_decay = 0.4 #decay rate
        self.epsilon_min = 0.1 #minimum epsilon value with decay
        self.gamma = 0.9 #discount factor
        self.memory = deque(maxlen=100_000)
        self.q_table = {}
        self.lr = 0.1 #learning rate
        self.actions = [0,1,2]

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
            #danger straight ahead
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            #danger on right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            #danger on left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            #move directions
            dir_l,dir_r,dir_u,dir_d,

            #food location
            game.food.x < game.head.x, #food left
            game.food.x > game.head.x, #food right'
            game.food.y < game.head.y, #food up
            game.food.y > game.head.y #food down
        ]

        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state,next_action, done):
        self.memory.append((state,action,reward,next_state,next_action,done))
    
    def train_short_memory(self, state, action, reward, next_state, next_action, done):
        state_str = str(state)
        next_state_str = str(next_state)

        if state_str not in self.q_table:
            self.q_table[state_str] = [0 for _ in range(len(self.actions))]
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = [0 for _ in range(len(self.actions))]

        current_q = self.q_table[state_str][action]
        future_q = self.q_table[next_state_str][next_action]

        if done:
            new_q = reward
        else:
            new_q = (1 - self.lr) * current_q + self.lr * (reward + self.gamma * future_q)
        
        self.q_table[state_str][action] = new_q
    
    def get_action(self, state):
        #self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        state_str = str(state)
        if random.randint(0,200) < self.epsilon:
            action = random.randint(0,2)
        else:
            if state_str in self.q_table:
                action = np.argmax(self.q_table[state_str])
            else:
                action = random.randint(0,2)
        return [1 if i == action else 0 for i in range(len(self.actions))]
    
    def train(self, total_games=1000, plot_every=100):
        scores = []
        mean_scores = []
        for i in range (total_games):
            game = SnakeGameAI()
            state = self.get_state(game)
            score=0
            action = self.get_action(state)
            while True:
                reward, done, score = game.play_step(action)
                action_index = action.index(1)
                next_state = self.get_state(game)
                next_action = self.get_action(next_state)
                next_action_index = next_action.index(1)

                self.train_short_memory(state, action_index, reward, next_state, next_action_index, done)
                self.remember(state, action, reward, next_state, next_action, done)
                state, action = next_state, next_action

                if done:
                    print(f'Game {self.n_games} Score: {score} Epsilon: {self.epsilon}')
                    self.n_games += 1
                    scores.append(score)
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                    game.reset()

                    if self.n_games == total_games:
                        mean_score = np.mean(scores[0:self.n_games])
                        mean_scores.append(mean_score)
                        plot(scores, mean_scores)
                    break

    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory
        for state, action, reward, next_state, next_action, done in mini_sample:
            self.train_short_memory(state, action, reward, next_state, next_action, done)
    
if __name__ == '__main__':
        agent = SARSAAgent()
        agent.train(200, 100)
        