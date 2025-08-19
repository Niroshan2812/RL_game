import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGame, Direction, Point

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon =0 # randomnes
        self.gamma = 0 # discount rate 
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()

        #model trainner


    def get_state(self, snake_game):
        pass
    def remember(self, state,action,reward,next_stage, done):
        pass
    def train_long_memory(self):
        pass
    def train_short_memory(self,state,action,reward,next_stage, done):
        pass
    def get_action(self,state):
        pass

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    snake_game = SnakeGame
    while True:
        #get old state 
        state_old = agent.get_state(snake_game)

        #get move
        final_move = agent.get_action(state_old)

        #performce move and get new state 
        reward, done, score = snake_game.play_step(final_move)
        state_new = agent.get_state(snake_game)

        #Train short memory
        agent.train_short_memory(state_old,final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory and plot result
            snake_game.reset()
            agent.n_game +=1
            agent.train_long_memory

            if score > record:
                record = score
            
            print ('Game: ', agent.n_games, 'Score: ', score, 'Record: ', record)

if __name__ == '__main__':
    train()
