import pygame
import random
import time
import socket

SOCKET_PORT = 1234
BUFFER_SIZE = 1024

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = None, None # none for now and will be updated when the game starts
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FPS = 120#360
MOVE_FILE_PATH = "../data/move.txt"
NETWORK_INPUT_FILE_PATH = "../data/gameStateData.txt"

# enable or disable the ability to view the game
viewGame = 0

# enable or disable keyboard input for user controlled paddle (controls left paddle)
userControlledPaddle = True

# enable or disable single player mode
singlePlayerMode = False

# initialize socket
clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientSocket.connect(('127.0.0.1', SOCKET_PORT))
print("connected to server")

def recvGameConstants(clientSocket):
    # Receive the game constants from the server
    data = clientSocket.recv(BUFFER_SIZE).decode('utf-8')
    print(f"Received game constants: {data}")
    # Parse the game constants (width, height)
    width, height = map(float, data.strip("{}").split(","))
    return width, height

def recvMove(clientSocket):
    # recieve move from the server
    data = clientSocket.recv(BUFFER_SIZE).decode('utf-8')
    # parse the move
    move = float(data.strip("{}"))
    return move

def sendGameState(client_socket, game_state):
    # Send the game state to the server
    client_socket.send(game_state.encode('utf-8'))

# read the game constants
while WIDTH is None or HEIGHT is None:
    WIDTH, HEIGHT = recvGameConstants(clientSocket)
    print(f"Game Constants - Width: {WIDTH}, Height: {HEIGHT}")

# player points
opponentPoints = 0
playerPoints = 0

# define the current player direction
# 0.5: no movement, greater than 0.5: up, less than 0.5: down
playerDirection = None

# same as above but previous direction (init to none for the first iteration)
prevPlayerDirection = None

# Create the screen
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

# Send the initial game state
#gameState = "{" + str(rightPaddle.y) + "," + str(leftPaddle.y) + "," + str(playerPoints) + "," + str(opponentPoints) + "," + str(ball.x) + "," + str(ball.y) + "," + str(ballVelocityX) + "," + str(ballVelocityY) + "}"
#sendGameState(clientSocket, gameState)

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


    # send the current game state
    # right paddle is the player paddle
    # the format: {playerPaddleY,opponentPaddleY,playerPoints,opponentPoints,ballX,ballY,ballVelocityX,ballVelocityY}
    gameState = "{" + str(rightPaddle.y) + "," + str(leftPaddle.y) + "," + str(playerPoints) + "," + str(opponentPoints) + "," + str(ball.x) + "," + str(ball.y) + "," + str(ballVelocityX) + "," + str(ballVelocityY) + "}"
    sendGameState(clientSocket, gameState)
    
    # Right paddle movement (player)
    # Take move input from the server and update the playerDirection variable
    # the format is {moveDirection} so we need to remove the curly braces and read the value
    # loop until the playerDirection is not None and it contains a float value
    playerDirection = recvMove(clientSocket)

    while playerDirection is None:
        try:
            playerDirection = recvMove(clientSocket)
        except:
            playerDirection = None


    # Player/agent (right) movement
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
    if userControlledPaddle:
        # opponent direction controlled by user
        # get input from keyboard
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and leftPaddle.y > 0:
            leftPaddle.y -= 5
        if keys[pygame.K_DOWN] and leftPaddle.y < HEIGHT - paddleHeight:
            leftPaddle.y += 5
    elif not singlePlayerMode:
        if leftPaddle.y + paddleHeight // 2 > ball.y: # + 20:
            leftPaddle.y -= 3
        # if the ball is below the paddle's center, move the paddle down
        elif leftPaddle.y + paddleHeight // 2 < ball.y: # - 20:
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

    # check render.txt file to check if the game should render
    with open("render.txt", "r") as file:
        content = file.read().strip()
        viewGame = content == "1"
        

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

clientSocket.close()
pygame.quit()