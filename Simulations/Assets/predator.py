import pygame
import math
import time
from Assets.prey import Prey
from shapely.geometry import LineString, Point

class Predator:
    WIN_WIDTH = 1000
    WIN_HEIGHT = 750
    def __init__(self, x, y):
        self.x = x
        self.y = y 
        self.speed = 1
        self.radius = 10
        self.originx, self.originy = x + self.radius, y + self.radius
        self.circle_surface = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        self.rotation = 0
        pygame.draw.circle(self.circle_surface, (100, 100, 100), (self.radius, self.radius), self.radius)
    
    def draw(self, window, prey):
        within_FoV = False
        line_surfaces = [pygame.Surface((self.WIN_WIDTH, self.WIN_HEIGHT), pygame.SRCALPHA) for _ in range(5)]
        _, end_points = self.line_collision(prey)
        for i in range (-2, 3):
            end_point_x, end_point_y = end_points[i+1][0], end_points[i+1][1]
            pygame.draw.line(line_surfaces[i+1], (120, 230, 100), (self.originx, self.originy), (end_point_x, end_point_y), 18) #for display purposes, set red
        for line_surface in line_surfaces:
            window.blit(line_surface, (0, 0))
        window.blit(self.circle_surface, (self.x, self.y))
        return within_FoV
    
    def line_collision(self, prey):
        end_points = []
        within_FoV = False
        for i in range (-2, 3):
            slope = math.radians((self.rotation + i*1.75))
            end_point_x = self.originx + 5000 * math.sin(slope)
            end_point_y = self.originy + 5000 * math.cos(slope)
                
            #check for colision
            prey_circle = Point(prey.originx, prey.originy).buffer(prey.radius)
            line = LineString([(self.originx, self.originy), (end_point_x, end_point_y)])
            intersection = prey_circle.intersection(line)
            if not intersection.is_empty:
                end_point_x, end_point_y = list(intersection.xy)[0][0], list(intersection.xy)[1][0]
                within_FoV = True
            end_points.append((end_point_x, end_point_y))
        return within_FoV, end_points
        
    def get_current_heading(self):
        slope = math.radians(self.rotation)
        end_point_x = self.originx + 10 * math.sin(slope)
        end_point_y = self.originy + 10 * math.cos(slope)
        return math.degrees(math.atan2(self.originy - end_point_y, self.originx - end_point_x))
    
    
    def move(self, rotation):
        #check not offscreen
        self.rotation = rotation
        new_origin_x = self.originx + self.speed * math.sin(math.radians(rotation))
        new_origin_y = self.originy + self.speed * math.cos(math.radians(rotation))
        self.originx, self.originy = new_origin_x, new_origin_y
        self.x, self.y = self.originx - self.radius, self.originy - self.radius
        
    def collides_with_partner(self, rand_angle, other_predator):
        this_predator = Predator(self.x, self.y) #this is where the predator would be if it moved in the direction specified by "rotation"
        #we are checking whether or not the predator will collide with another predator if it moved like so
        this_predator.move(rand_angle)
        collision = math.sqrt((this_predator.originx - other_predator.originx)**2 + (this_predator.originy - other_predator.originy)**2) <= 2 * self.radius
        return collision
    
    def collides_with_prey(self, preyOriginx, preyOriginy):
        collision = math.sqrt((self.originx - preyOriginx)**2 + (self.originy - preyOriginy)**2) <= 2 * self.radius
        return collision
    
if __name__ == "__main__":
    pygame.init()
    pygame.font.init()
    pred = Predator(300, 500)
    window = pygame.display.set_mode((pred.WIN_WIDTH, pred.WIN_HEIGHT))
    prey = Prey(100, 200)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                quit()
                
        rect_surface = pygame.Surface((pred.WIN_WIDTH, pred.WIN_HEIGHT)) #rectangle surface
        pygame.draw.rect(rect_surface, (100, 150, 100), (0, 0, pred.WIN_WIDTH, pred.WIN_HEIGHT))
        window.blit(rect_surface, (0, 0))
        
        pred.rotation += 1
        pred.rotation = pred.rotation % 360
        pred.draw(window, prey)
        prey.draw(window)
        pygame.display.update()
