import math
import pygame
import sys
import os
import neat
from abc import ABC

WIDTH = 1000
HEIGHT = 700
CAPTION = "AI Racing"

INNER_WALLS = [(656, 469), (319, 502), (177, 395), (104, 439), (86, 542), (231, 613), (891, 645),
               (897, 580), (698, 556), (779, 346), (916, 300), (938, 114), (836, 73), (750, 90),
               (674, 207), (327, 203), (124, 94), (114, 105), (317, 218), (700, 219), (682, 470)]
OUTER_WALLS = [(41, 42), (202, 37), (339, 152), (627, 148), (698, 34), (833, 13), (988, 84),
               (977, 339), (804, 409), (767, 510), (960, 525), (945, 695), (209, 672), (28, 589),
               (35, 400), (176, 308), (336, 446), (620, 408), (621, 275), (291, 306), (92, 238)]

WALL_WIDTH = 3
LINE_WIDTH = 1

WALL_COLOR = (0, 0, 0)
BACK_COLOR = (145, 145, 145)
LINE_COLOR = (255, 255, 56)
TEXT_COLOR = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Initialize pygame and create window
pygame.init()
os.environ['SDL_VIDEO_CENTERED'] = '1'
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(CAPTION)

CLOCK = pygame.time.Clock()

WALLS = set()

BLUE_CAR = pygame.image.load("imgs/blue_car.png").convert_alpha()
GREEN_CAR = pygame.image.load("imgs/green_car.png").convert_alpha()
RED_CAR = pygame.image.load("imgs/red_car.png").convert_alpha()

generation = -1
total = 0


# Converts Vector2 to a Vector2 of ints
def int_vector2(v):
    return pygame.Vector2(int(v.x), int(v.y))


# Clamps number between a minimum and maximum
def clamp(x, mn, mx):
    if x < mn:
        x = mn
    elif x > mx:
        x = mx
    return x


# Returns distance between two vertexes(Vector2)
def dist(v1, v2):
    return math.sqrt((v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2)


# Sends ray from vertex start at angle rot and returns closest intersection with wall
def ray_cast(start, rot):
    found = False
    v = pygame.Vector2(0, 0)
    v2 = pygame.Vector2(0, 0)
    for wall in WALLS:
        tana = math.tan(rot)
        tanb = math.tan(math.atan2(wall.end.y - wall.start.y, wall.end.x - wall.start.x))
        if tana == tanb or tana == -tanb:
            continue
        v.x = (wall.start.y - start.y + start.x * tana - wall.start.x * tanb) / (tana - tanb)
        v.y = wall.start.y + (v.x - wall.start.x) * tanb
        cond = (wall.start.x <= v.x <= wall.end.x or wall.start.x >= v.x >= wall.end.x)
        cond = cond and (wall.start.y <= v.y <= wall.end.y or wall.start.y >= v.y >= wall.end.y)
        orcond = (math.radians(-180) <= rot <= math.radians(-90)) and (start.x >= v.x and start.y >= v.y)
        orcond = orcond or (math.radians(-90) <= rot <= math.radians(0)) and (start.x <= v.x and start.y >= v.y)
        orcond = orcond or (math.radians(0) <= rot <= math.radians(90)) and (start.x <= v.x and start.y <= v.y)
        orcond = orcond or (math.radians(90) <= rot <= math.radians(180)) and (start.x >= v.x and start.y <= v.y)
        cond = cond and orcond
        if cond:
            if (not found) or (dist(start, v2) > dist(start, v)):
                found = True
                v2.x = v.x
                v2.y = v.y
    return v2


# Represents the text shown on screen
class Text:

    def __init__(self):
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.string = ""
        self.text = self.font.render(self.string, True, TEXT_COLOR)
        self.rect = self.text.get_rect()
        self.rect.center = (75, 20)

    def draw(self):
        self.string = 'Generation: ' + str(generation) + ' FPS:' + str(int(CLOCK.get_fps()))
        self.text = self.font.render(self.string, True, TEXT_COLOR)
        SCREEN.blit(self.text, self.rect)


# Initialize text
TEXT = Text()


# Represents a line object
class Line:

    def __init__(self, xy1, xy2, width, c):
        self.start = int_vector2(xy1)
        self.end = int_vector2(xy2)
        self.color = c
        self.width = width

    # Returns length of line
    def len(self):
        return math.sqrt((self.start.x - self.end.x)**2 + (self.start.y - self.end.y)**2)

    def draw(self):
        pygame.draw.line(SCREEN, self.color, self.start, self.end, self.width)


# Represents an abstract car
class Car(ABC):

    def __init__(self):
        self.rotation = 0
        self.wheel_rotation = 0
        self.pos = pygame.Vector2(490, 180)
        self.length = 20
        self.width = 10
        self.shoLines = False
        self.lines = []
        self.sprite = RED_CAR
        self.speed = 5
        self.time = 0
        self.avg_speed = 0
        self.crashed = False
        for i in range(5):
            self.lines.append(Line(self.pos, self.pos, LINE_WIDTH, LINE_COLOR))
        self.setup()

    def draw(self):
        #for line in self.lines:
        #    line.draw()
        #    pygame.draw.circle(SCREEN, BLUE, (int(line.end.x), int(line.end.y)), 3)
        rotated_sprite = pygame.transform.rotate(self.sprite, math.degrees(2*math.pi - self.rotation))
        sprite_rect = rotated_sprite.get_rect()
        sprite_rect.center = self.pos
        SCREEN.blit(rotated_sprite, sprite_rect)

    # Move car and send rays in front of it in order to detect walls
    def move(self, dt):

        # If it already crashed it no longer moves
        if self.crashed:
            return

        # Move car based on input given by the neural network or player
        # by computing positions of front and back wheels
        # and approximating the car's position based on them
        (d_speed, d_wheel_rotation) = self.get_input()
        self.speed += d_speed * 0.07 * dt
        self.wheel_rotation += d_wheel_rotation * 0.0045 * dt
        self.speed -= 0.02 * dt
        if self.wheel_rotation > 0:
            self.wheel_rotation = max(0.0, self.wheel_rotation - 0.004 * dt)
        if self.wheel_rotation < 0:
            self.wheel_rotation = min(0.0, self.wheel_rotation + 0.004 * dt)

        self.speed = clamp(self.speed, 10, 40)
        if self.speed == 0:
            self.crash()
            return
        self.wheel_rotation = clamp(self.wheel_rotation, - 0.75, 0.75)
        dt *= 0.005
        rot = pygame.Vector2(math.cos(self.rotation), math.sin(self.rotation))
        wheel_rot = pygame.Vector2(math.cos(self.rotation + self.wheel_rotation), math.sin(self.rotation + self.wheel_rotation))

        front_wheel = self.pos + self.length / 2 * rot
        back_wheel = self.pos - self.length / 2 * rot
        front_wheel += self.speed * dt * wheel_rot
        back_wheel += self.speed * dt * rot
        self.pos = (front_wheel + back_wheel) / 2
        self.rotation = math.atan2((front_wheel.y - back_wheel.y), (front_wheel.x - back_wheel.x))

        # Compute the car's average speed
        self.time += dt
        self.avg_speed = ((self.time - dt) * self.avg_speed + self.speed) / self.time

        # Send rays in front of car in order to detect walls
        rot = self.rotation - math.radians(135)
        for line in self.lines:
            rot += math.radians(45)
            if rot > math.pi:
                rot -= math.pi * 2
            if rot < -math.pi:
                rot += math.pi * 2
            line.start = self.pos
            line.end = ray_cast(self.pos, rot)
            if line.len() < 5:
                self.crash()

        self.end_move(dt)


# Represents a car controlled by a player
class PlayerCar(Car):

    def setup(self):
        self.sprite = BLUE_CAR

    # Gets input from the player
    # and returns it
    @staticmethod
    def get_input(self):
        keys = pygame.key.get_pressed()
        d_speed = 0
        if keys[pygame.K_w]:
            d_speed += 1
        if keys[pygame.K_s]:
            d_speed -= 1
        d_wheel_rotation = 0
        if keys[pygame.K_a]:
            d_wheel_rotation -= 1
        if keys[pygame.K_d]:
            d_wheel_rotation += 1
        return d_speed, d_wheel_rotation

    # Handles the car crashing
    def crash(self):
        self.crashed = True

    def end_move(self):
        pass


# Represents a car controlled by NEAT
class LearningCar(Car):

    def setup(self):
        self.sprite = GREEN_CAR

    # Gets input from the neural network
    # and returns it
    def get_input(self):
        (speed, wheel_rotation) = self.net.activate((self.speed, self.wheel_rotation, self.lines[0].len(), self.lines[1].len(), self.lines[2].len(), self.lines[3].len(), self.lines[4].len()))
        if speed > 0.5:
            d_speed = 2
        elif speed < -0.5:
            d_speed = -2
        else:
            d_speed = 0
        if wheel_rotation > 0.5:
            d_wheel_rotation = 2
        elif wheel_rotation < -0.5:
            d_wheel_rotation = -2
        else:
            d_wheel_rotation = 0
        return d_speed, d_wheel_rotation

    # Handles the car crashing
    def crash(self):
        if self.crashed:
            return
        self.crashed = True
        self.sprite = RED_CAR
        self.genome.fitness -= 10

        global total
        total -= 1
        print(total)

    def end_move(self, dt):
        self.genome.fitness += self.avg_speed**2 * dt / 10000


# Simulates the current generation of cars
# to compute their fitness
def find_fitness(genomes, config):

    global generation
    generation += 1

    # Initialize current generation of cars
    # and put it in a list
    cars = []
    for genome_id, genome in genomes:
        car = LearningCar()
        genome.fitness = 0
        car.net = neat.nn.FeedForwardNetwork.create(genome, config)
        car.genome = genome
        cars.append(car)

    # Run simulation and draw everything to screen
    running = True
    global total
    total = len(genomes)
    print(total)
    while running and total > 0:
        SCREEN.fill(BACK_COLOR)
        dt = CLOCK.tick(360)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done()
            if event.type == pygame.MOUSEBUTTONDOWN:
                b = event.button
                if (b == 1):
                    running = False

        for wall in WALLS:
            wall.draw()

        for car in cars:
            car.move(dt)
            car.draw()

        TEXT.draw()

        pygame.display.update()


# Exit the program
def done():
    pygame.quit()
    sys.exit()


# Add a cycle of walls to WALLS from a list of vertexes(tuples)
def add_wall_cycle(input_walls):
    has_prev = False
    prev = pygame.Vector2(0, 0)
    init = pygame.Vector2(0, 0)
    for tup in input_walls:
        xy = pygame.Vector2(tup[0], tup[1])
        if has_prev:
            WALLS.add(Line(prev, xy, WALL_WIDTH, WALL_COLOR))
            prev = xy
        else:
            has_prev = True
            init = xy
            prev = xy
    WALLS.add(Line(prev, init, WALL_WIDTH, WALL_COLOR))


# Import walls
add_wall_cycle(INNER_WALLS)
add_wall_cycle(OUTER_WALLS)


# Train a neural network to drive a car using NEAT
def run_neat(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)

    #p.add_reporter(neat.StdOutReporter(True))
    #stats = neat.StatisticsReporter()
    #p.add_reporter(stats)

    winner = p.run(find_fitness, 50)


if __name__ == "__main__":
    # Determine config path file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run_neat(config_path)

done()
