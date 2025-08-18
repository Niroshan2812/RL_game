import pygame;
import random;

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class SnakeGame:
    def __init__(self, w=640,h=480):
        self.w = w
        self.h=h

        #game display envirement 
        self.display =  pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

        #print ("initial")



if __name__ =='__main__':
    game = SnakeGame()