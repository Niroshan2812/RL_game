import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGame, Direction, Point
from model import Linner_QNet, Qtrainner
from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon =0 # randomnes
        self.gamma = 0.9 # discount rate 
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model =  Linner_QNet(11, 256, 3)
        self.trainer = Qtrainner(self.model, lr = LR, gamma = self.gamma)

        #model trainner


    def get_state(self, snake_game):
        head = snake_game.head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = snake_game.direction == Direction.LEFT
        dir_r = snake_game.direction == Direction.RIGHT
        dir_u = snake_game.direction == Direction.UP
        dir_d = snake_game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and snake_game.is_collision(point_r)) or
            (dir_l and snake_game.is_collision(point_l)) or
            (dir_u and snake_game.is_collision(point_u)) or
            (dir_d and snake_game.is_collision(point_d)),

            # danger right
            (dir_u and snake_game.is_collision(point_r)) or
            (dir_d and snake_game.is_collision(point_l)) or
            (dir_l and snake_game.is_collision(point_u)) or
            (dir_r and snake_game.is_collision(point_d)),

            # danger left
            (dir_d and snake_game.is_collision(point_r)) or
            (dir_u and snake_game.is_collision(point_l)) or
            (dir_r and snake_game.is_collision(point_u)) or
            (dir_l and snake_game.is_collision(point_d)),

            # move direction 
            dir_l,
            dir_r,
            dir_u, 
            dir_d,

            # food location
            # if game food is smaller than game head then {x}
            snake_game.food.x < snake_game.head.x,  # {x} => food left
            snake_game.food.x > snake_game.head.x,  # {x} => food right
            snake_game.food.y < snake_game.head.y,  # {x} => food up
            snake_game.food.y > snake_game.head.y   # {x} => food down
        ]
        return np.array(state,dtype = int)
    


    def remember(self, state,action,reward,next_stage, done):
        self.memory.append((state, action, reward, next_stage, done))

    def train_long_memory(self):
        #Check how many sample in memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, min(BATCH_SIZE, len(self.memory))) # return list of tuples 
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_stages, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_stages, dones)



    def train_short_memory(self,state,action,reward,next_stage, done):
        self.trainer.train_step(state, action, reward, next_stage, done)

    def get_action(self,state):
        # random move
        self.epsilon = 80 - self.n_games
        final_move= [0,0,0]
        if random.randint(0,200)<self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction  = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move [move]= 1
        return final_move



def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    snake_game = SnakeGame()
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
            agent.n_games +=1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            print ('Game: ', agent.n_games, 'Score: ', score, 'Record: ', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
