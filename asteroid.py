import os
import time
import random
import msvcrt  # For Windows keypress detection (use 'getch' library for Linux/Mac)

# Game settings
WIDTH = 20
HEIGHT = 10
SHIP = 'S'
ASTEROID = '*'
BULLET = '|'
EMPTY = ' '

# Initial game state
ship_x = WIDTH // 2
asteroids = []
bullets = []
score = 0
game_over = False

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_grid():
    grid = [[EMPTY for _ in range(WIDTH)] for _ in range(HEIGHT)]
    # Place ship
    grid[HEIGHT - 1][ship_x] = SHIP
    # Place asteroids
    for ax, ay in asteroids:
        if 0 <= ay < HEIGHT and 0 <= ax < WIDTH:
            grid[ay][ax] = ASTEROID
    # Place bullets
    for bx, by in bullets:
        if 0 <= by < HEIGHT and 0 <= bx < WIDTH:
            grid[by][bx] = BULLET
    # Print grid
    clear_screen()
    print(f"Score: {score}")
    for row in grid:
        print(''.join(row))
    print("\nControls: A (left), D (right), Space (shoot), Q (quit)")

def move_asteroids():
    global asteroids
    asteroids = [(x, y + 1) for x, y in asteroids if y + 1 < HEIGHT]

def spawn_asteroid():
    if random.random() < 0.3:  # 30% chance to spawn an asteroid each frame
        asteroids.append((random.randint(0, WIDTH - 1), 0))

def move_bullets():
    global bullets
    bullets = [(x, y - 1) for x, y in bullets if y - 1 >= 0]

def check_collisions():
    global score, game_over
    # Bullet-asteroid collisions
    new_asteroids = []
    for ax, ay in asteroids:
        hit = False
        for bx, by in bullets:
            if ax == bx and ay == by:
                score += 1
                hit = True
                bullets.remove((bx, by))
                break
        if not hit:
            new_asteroids.append((ax, ay))
    # Ship-asteroid collision
    for ax, ay in asteroids:
        if ay == HEIGHT - 1 and ax == ship_x:
            game_over = True
    return new_asteroids

# Main game loop
while not game_over:
    draw_grid()
    spawn_asteroid()
    move_asteroids()
    move_bullets()
    asteroids = check_collisions()

    # Handle input (non-blocking)
    if msvcrt.kbhit():
        key = msvcrt.getch().decode('utf-8').lower()
        if key == 'a' and ship_x > 0:
            ship_x -= 1
        elif key == 'd' and ship_x < WIDTH - 1:
            ship_x += 1
        elif key == ' ':
            bullets.append((ship_x, HEIGHT - 2))
        elif key == 'q':
            game_over = True

    time.sleep(0.2)  # Control game speed

clear_screen()
print(f"Game Over! Final Score: {score}")