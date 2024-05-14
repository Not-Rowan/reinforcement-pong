import pygame
import random
import time

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = None, None # none for now and will be updated when the game starts
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FPS = 360
MOVE_FILE_PATH = "../data/move.txt"
NETWORK_INPUT_FILE_PATH = "../data/gameStateData.txt"

# enable or disable the ability to view the game
viewGame = False

# enable or disable keyboard input for user controlled paddle
userControlledPaddle = False

# enable or disable single player mode
singlePlayerMode = False

# read game constants from the gameConstants.txt file {width,height}
while WIDTH is None or HEIGHT is None:
    with open("../data/gameConstants.txt", "r") as f:
        gameConstants = f.read()
        if gameConstants != "":
            gameConstants = gameConstants.replace("{", "").replace("}", "").split(",")
            WIDTH = int(gameConstants[0])
            HEIGHT = int(gameConstants[1])
        else:
            print("Error reading game constants file")

# player points
opponentPoints = 0
playerPoints = 0

# define the current player direction
# 0.5: no movement, greater than 0.5: up, less than 0.5: down
playerDirection = None

# same as above but previous direction (init to none for the first iteration)
prevPlayerDirection = None

# Create the screen
if viewGame:
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong")
    clock = pygame.time.Clock()

# Define the game objects
paddleWidth = 10
paddleHeight = 100
ballSize = 10

# Paddle initialization
if not singlePlayerMode:
    leftPaddle = pygame.Rect(50, HEIGHT // 2 - paddleHeight // 2, paddleWidth, paddleHeight)
else:
    # fill entire left side of the screen with a paddle
    leftPaddle = pygame.Rect(0, 0, paddleWidth, HEIGHT)
rightPaddle = pygame.Rect(WIDTH - 50 - paddleWidth, HEIGHT // 2 - paddleHeight // 2, paddleWidth, paddleHeight)

# Ball initialization
ball = pygame.Rect(WIDTH // 2 - ballSize // 2, HEIGHT // 2 - ballSize // 2, ballSize, ballSize)
ballVelocityX = random.choice([-3, 3])
ballVelocityY = random.choice([-3, 3])

# Write the initial game state to the gameStateData file
with open(NETWORK_INPUT_FILE_PATH, "w") as f:
    f.write("{" + str(rightPaddle.y) + "," + str(leftPaddle.y) + "," + str(playerPoints) + "," + str(opponentPoints) + "," + str(ball.x) + "," + str(ball.y) + "," + str(ballVelocityX) + "," + str(ballVelocityY) + "}")

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Ball movement
    ball.x += ballVelocityX
    ball.y += ballVelocityY

    # Ball collision with paddles (handle collisions with sides and top)
    # if the ball hits the top or bottom of the paddles, make it go a bit faster in the y axis. if it hits the middle it should go forward straight
    if ball.colliderect(leftPaddle) and not singlePlayerMode:
        # Reverse horizontal velocity and adjust vertical velocity based on the position of the ball on the paddle
        ballVelocityX = abs(ballVelocityX)  # Ensure the ball always moves to the right after collision
        ballVelocityY = int((ball.centery - leftPaddle.centery) / (paddleHeight / 2) * 5)  # Adjust vertical velocity between -5 and 5
    elif ball.colliderect(rightPaddle):
        # Reverse horizontal velocity and adjust vertical velocity based on the position of the ball on the paddle
        ballVelocityX = -abs(ballVelocityX)  # Ensure the ball always moves to the left after collision
        ballVelocityY = int((ball.centery - rightPaddle.centery) / (paddleHeight / 2) * 5)  # Adjust vertical velocity between -5 and 5
    elif singlePlayerMode and ball.colliderect(leftPaddle):
        ballVelocityX *= -1
    

    # Ball collision with walls
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ballVelocityY *= -1

    # Ball out of bounds
    if ball.left <= 0 and not singlePlayerMode:
        # player scores
        playerPoints += 1
        ball.center = (WIDTH // 2, HEIGHT // 2)
        ballVelocityX = random.choice([-3, 3])
        ballVelocityY = random.choice([-3, 3])
    elif ball.right >= WIDTH:
        # opponent scores
        opponentPoints += 1
        ball.center = (WIDTH // 2, HEIGHT // 2)
        ballVelocityX = random.choice([-3, 3])
        ballVelocityY = random.choice([-3, 3])


    # write the current game state / game data to the gameStateData file
    # right paddle is the player paddle
    # the format: {playerPaddleY,opponentPaddleY,playerPoints,opponentPoints,ballX,ballY,ballVelocityX,ballVelocityY}
    with open(NETWORK_INPUT_FILE_PATH, "w") as f:
        f.write("{" + str(rightPaddle.y) + "," + str(leftPaddle.y) + "," + str(playerPoints) + "," + str(opponentPoints) + "," + str(ball.x) + "," + str(ball.y) + "," + str(ballVelocityX) + "," + str(ballVelocityY) + "}")
    
    # Right paddle movement (player)
    # Take input from move.txt and update the playerDirection variable
    # the format of the file is {moveDirection} so we need to remove the curly braces and read the value
    # loop until the playerDirection is not None and it contains a float value

    # get initial player value
    with open(MOVE_FILE_PATH, "r") as f:
        playerDirection = f.read()
        if playerDirection != "":
            try:
                playerDirection = float(playerDirection.replace("{", "").replace("}", ""))
            except:
                playerDirection = None
        else:
            playerDirection = None

    # then make sure a value has been read
    # python sucks. no do while loop
    while playerDirection is None or not isinstance(playerDirection, float):
        with open(MOVE_FILE_PATH, "r") as f:
            playerDirection = f.read()
            if playerDirection != "":
                playerDirection = float(playerDirection.replace("{", "").replace("}", ""))
            else:
                playerDirection = None


    if userControlledPaddle:
        # player direction controlled by user
        # get input from keyboard
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and rightPaddle.y > 0:
            rightPaddle.y -= 5
        if keys[pygame.K_DOWN] and rightPaddle.y < HEIGHT - paddleHeight:
            rightPaddle.y += 5
    else:
        # before moving, make sure the player would be able to move the paddle in the next iteration
        # if the playerDirection is greater than 0.5, move the paddle up
        if playerDirection > 0.5 and rightPaddle.y > 0:
            rightPaddle.y -= 5
        # if the playerDirection is less than 0.5, move the paddle down
        elif playerDirection < 0.5 and rightPaddle.y < HEIGHT - paddleHeight:
            rightPaddle.y += 5
        # update the prevPlayerDirection variable
        prevPlayerDirection = playerDirection


    # Left paddle movement (opponent)
    # if the ball is above the paddle's center, move the paddle up
    if not singlePlayerMode:
        if leftPaddle.y + paddleHeight // 2 > ball.y + 20:
            leftPaddle.y -= 3
        # if the ball is below the paddle's center, move the paddle down
        elif leftPaddle.y + paddleHeight // 2 < ball.y - 20:
            leftPaddle.y += 3

    # prevent paddles from going out of bounds
    # also prevent the left paddle from going right to the edge so we dont have the ball constantly bouncing back and forth
    if rightPaddle.y <= 0:
        rightPaddle.y = 0
    if rightPaddle.y >= HEIGHT - paddleHeight:
        rightPaddle.y = HEIGHT - paddleHeight
    if leftPaddle.y <= 10:
        leftPaddle.y = 10
    if leftPaddle.y >= HEIGHT - paddleHeight - 10:
        leftPaddle.y = HEIGHT - paddleHeight - 10

    # Render
    if viewGame:
        screen.fill(BLACK)
        pygame.draw.rect(screen, WHITE, leftPaddle)
        pygame.draw.rect(screen, WHITE, rightPaddle)
        pygame.draw.ellipse(screen, WHITE, ball)
        pygame.draw.aaline(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))

        font = pygame.font.SysFont(None, 50)
        p2_text = font.render(str(opponentPoints), True, WHITE)
        p1_text = font.render(str(playerPoints), True, WHITE)
        screen.blit(p2_text, (WIDTH // 4, 50))
        screen.blit(p1_text, (WIDTH * 3 // 4, 50))

        pygame.display.flip()
        clock.tick(FPS)

pygame.quit()