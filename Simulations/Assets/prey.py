import pygame
import math

class Prey:
    WIN_WIDTH = 1000
    WIN_HEIGHT = 750
    def __init__(self, x, y):
        self.x = x
        self.y = y 
        self.speed = 1.5
        self.radius = 10
        self.originx, self.originy = x + self.radius, y + self.radius
        self.circle_surface = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        self.rotation = 0
        pygame.draw.circle(self.circle_surface, (200, 50, 50), (self.radius, self.radius), self.radius)
    
    def draw(self, window):
        window.blit(self.circle_surface, (self.x, self.y))
        
    def move(self, rotation):
        #check not offscreen
        new_x = self.x + self.speed * math.sin(rotation)
        new_y = self.y + self.speed * math.cos(rotation)
        if new_x > 0 and new_x + 2 * self.radius < self.WIN_WIDTH and new_y > 0 and new_y + 2 * self.radius < self.WIN_HEIGHT:
            self.x, self.y = new_x, new_y
            self.originx, self.originy = self.x + self.radius, self.y + self.radius
    
    def get_current_heading(self):
        slope = math.radians(self.rotation)
        end_point_x = self.originx + 10 * math.sin(slope)
        end_point_y = self.originy + 10 * math.cos(slope)
        return math.degrees(math.atan2(self.originy - end_point_y, self.originx - end_point_x))
    
    def get_mask(self):
        return pygame.mask.from_surface(self.circle_surface)