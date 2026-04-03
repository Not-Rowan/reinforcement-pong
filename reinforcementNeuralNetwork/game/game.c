#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <SDL2/SDL.h>

#define PORT1 1234
#define PORT2 1235

#define PADDLE_WIDTH 10
#define PADDLE_HEIGHT 100
#define BALL_SIZE 10

#define USER_CONTROLLED_PADDLE 0
#define DUELING_AGENTS 1
#define FPS 120

int width = 0;
int height = 0;

typedef struct Ball {
    // allow ball vars and SDL_FRect vars to share the same memory space
    union {
        struct {float x, y, w, h;};
        SDL_FRect rect;
    };
    float velX, velY;
} Ball;

typedef struct Paddle {
    // allow paddle vars and SDL_FRect vars to share the same memory space
    union {
        struct {float x, y, w, h;};
        SDL_FRect rect;
    };
} Paddle;

/*
 ** Description
 * Creates a window to be displayed on the screen
 ** Parameters
 * windowTitle: the title that is displayed on the menu bar of the window
 * windowPosX: the x position on the screen to display the window
 * windowPosY: the y position on the screen to display the window
 * windowSizeX: the x size of the displayed window
 * windowSizeY: the y size of the displayed window
 * flags: additional flags for things like look of window, functionality, etc...
 ** Return Value
 * window: returns a pointer to the window that has been displayed or null if error
 */
SDL_Window *createWindow(const char *windowTitle, int windowPosX, int windowPosY, int windowSizeX, int windowSizeY, int flags) {
    if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
        fprintf(stderr, "Failed to initialize the SDL2 library\n");
        return NULL;
    }

    SDL_Window *window = SDL_CreateWindow(windowTitle, windowPosX, windowPosY, windowSizeX, windowSizeY, flags);

    if (!window) {
        fprintf(stderr, "Failed to create window\n");
        return NULL;
    }

    return window;
}

int connectToServer(int port) {
    int clientSock, status;
    if ((clientSock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("Socket creation error\n");
        return -1;
    }

    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);

    if (inet_pton(AF_INET, "127.0.0.1", &serverAddr.sin_addr) <= 0) {
        printf("Invalid address/ Address not supported\n");
        return -1;
    }

    if ((status = connect(clientSock, (struct sockaddr*)&serverAddr, sizeof(serverAddr))) < 0) {
        printf("\nConnection Failed \n");
        return -1;
    }

    return clientSock;
}

int recvGameConstants(int *width, int *height, int clientSock) {
    char constantBuff[50];
    int retVal;
    if ((retVal = recv(clientSock, constantBuff, sizeof(constantBuff)-1, 0)) <= 0) {
        printf("Error receiving game constants\n");
        return -1;
    }
    constantBuff[retVal] = '\0';


    if ((retVal = sscanf(constantBuff, "{%d, %d}", width, height)) != 2) {
        printf("Could not extract game constants\n");
        return -1;
    }

    return 0;
}

void initGameObjs(Ball *ball, Paddle *leftPaddle, Paddle *rightPaddle) {
    // ball
    ball->x = width/2;
    ball->y = height/2;
    ball->w = BALL_SIZE;
    ball->h = BALL_SIZE;
    ball->velX = (rand() % 2 == 0) ? 3 : -3;
    ball->velY = (rand() % 2 == 0) ? 3 : -3;

    // left paddle
    leftPaddle->x = 50;
    leftPaddle->y = height/2 - PADDLE_HEIGHT/2;
    leftPaddle->w = PADDLE_WIDTH;
    leftPaddle->h = PADDLE_HEIGHT;

    // right paddle
    rightPaddle->x = width-50-PADDLE_WIDTH;
    rightPaddle->y = height/2 - PADDLE_HEIGHT/2;
    rightPaddle->w = PADDLE_WIDTH;
    rightPaddle->h = PADDLE_HEIGHT;
}

// returns 1 on collision, 0 otherwise
int collideRect(SDL_FRect *rect1, SDL_FRect *rect2) {
    // rect1 inside rect2 (left, right, top, bottom impacts respectively)
    if (rect1->x + rect1->w > rect2->x && 
        rect1->x < rect2->x + rect2->w && 
        rect1->y + rect1->h > rect2->y && 
        rect1->y < rect2->y + rect2->h
        ) return 1;

    return 0;
}

int main() {
    int renderGame = 0;
    int leftPts = 0, rightPts = 0;

    int clientSock1 = connectToServer(PORT1);
    if (clientSock1 < 0) {
        return -1;
    }

    int clientSock2 = connectToServer(PORT2);
    if (clientSock2 < 0) {
        return -1;
    }

    int retVal = recvGameConstants(&width, &height, clientSock1);
    if (retVal < 0) {
        return -1;
    }

    // not really needed. just allows the second client to continue the process
    retVal = recvGameConstants(&width, &height, clientSock2);
    if (retVal < 0) {
        return -1;
    }

    printf("game constants: %dx%d\n", width, height);

    // create window and renderer
    SDL_Window *window = createWindow("pong", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    srand(time(NULL)); // init random number stuff


    int gameRunning = 1;

    // init game objs
    Ball ball;
    Paddle leftPaddle;
    Paddle rightPaddle;
    initGameObjs(&ball, &leftPaddle, &rightPaddle);

    // init background stuff
    SDL_FRect line;
    line.x = width/2;
    line.y = 0;
    line.w = 3;
    line.h = height;

    // get pointer to internal auto-updating key state array
    const Uint8 *keyState = SDL_GetKeyboardState(NULL);

    int framesSinceFileRead = 0;

    while (gameRunning) {
        SDL_Event e;
        while(SDL_PollEvent(&e) > 0) {
            if (e.type == SDL_QUIT) {
                gameRunning = 0;
            }
        }

        //
        // Game Updates
        //

        ball.x += ball.velX;
        ball.y += ball.velY;

        // ball collision with paddles
        if (collideRect(&ball.rect, &leftPaddle.rect)) {
            // reverse horizontal velocity and adjust vertical velocity based on the position of the ball on the paddle
            ball.velX = fabs(ball.velX); // force ball to go right
            ball.velY = ((ball.y+(ball.h/2)) - (leftPaddle.y+(leftPaddle.h/2))) / (PADDLE_HEIGHT/2) * 5; // adjust vertical velocity between -5 and 5
        } else if (collideRect(&ball.rect, &rightPaddle.rect)) {
            ball.velX = -fabs(ball.velX); // force ball to go left
            ball.velY = ((ball.y+(ball.h/2)) - (rightPaddle.y+(rightPaddle.h/2))) / (PADDLE_HEIGHT/2) * 5; // adjust vertical velocity between -5 and 5
        }

        // ball collision with walls
        if (ball.y <= 0 || ball.y+ball.h >= height) {
            ball.velY *= -1;
        }

        // ball out of bounds
        if (ball.x <= 0) {
            // right scores & reset
            rightPts++;
            ball.x = width/2;
            ball.y = width/2;
            ball.velX = (rand() % 2 == 0) ? 3 : -3;
            ball.velY = (rand() % 2 == 0) ? 3 : -3;
        } else if (ball.x >= width) {
            // left scores & reset
            leftPts++;
            ball.x = width/2;
            ball.y = width/2;
            ball.velX = (rand() % 2 == 0) ? 3 : -3;
            ball.velY = (rand() % 2 == 0) ? 3 : -3;
        }

        // broadcast game state
        // right paddle is the player paddle. left paddle is the opponent paddle
        // the format: {rightPaddleY,leftPaddleY,playerPoints,opponentPoints,ballX,ballY,ballVelocityX,ballVelocityY}
        char gameState[100];
        int len = snprintf(gameState, sizeof(gameState), "{%f,%f,%d,%d,%f,%f,%f,%f}", rightPaddle.y, leftPaddle.y, rightPts, leftPts, ball.x, ball.y, ball.velX, ball.velY);
        if (len >= sizeof(gameState)) {
            send(clientSock1, NULL, 0, 0); // Send nothing. I don't wanna handle cases like this rn
            send(clientSock2, NULL, 0, 0);
        } else {
            send(clientSock1, gameState, len, 0);
            send(clientSock2, gameState, len, 0);
        }

        // recieve right paddle (player) direction
        char playerDirectionBuf[11];
        float playerDirection = -1;
        int bytesRead = 0;
        do {
            bytesRead = recv(clientSock1, playerDirectionBuf, sizeof(playerDirectionBuf)-1, 0);
            if (bytesRead > 0) {
                playerDirectionBuf[10] = '\0';
                sscanf(playerDirectionBuf, "{%f}", &playerDirection);
            } else {
                printf("Error receiving player direction\n");
                gameRunning = 0;
            }
        } while (bytesRead <= 0 && gameRunning);

        // right paddle movement
        if (playerDirection > 0.5f && rightPaddle.y > 0) {
            rightPaddle.y -= 5;
        } else if (playerDirection < 0.5f && rightPaddle.y < height - PADDLE_HEIGHT) {
            rightPaddle.y += 5;
        }

        // left paddle movement
        if (DUELING_AGENTS) {
            // recieve left paddle (opponent) direction
            char opponentDirectionBuf[11];
            float opponentDirection = -1;
            bytesRead = 0;
            do {
                bytesRead = recv(clientSock2, opponentDirectionBuf, sizeof(opponentDirectionBuf)-1, 0);
                if (bytesRead > 0) {
                    opponentDirectionBuf[10] = '\0';
                    sscanf(opponentDirectionBuf, "{%f}", &opponentDirection);
                } else {
                    printf("Error receiving opponent direction\n");
                    gameRunning = 0;
                }
            } while (bytesRead <= 0 && gameRunning);

            // left paddle movement
            if (opponentDirection > 0.5f && leftPaddle.y > 0) {
                leftPaddle.y -= 5;
            } else if (opponentDirection < 0.5f && leftPaddle.y < height - PADDLE_HEIGHT) {
                leftPaddle.y += 5;
            }
        } else if (USER_CONTROLLED_PADDLE) {
            if (keyState[SDL_SCANCODE_UP] && leftPaddle.y > 0) leftPaddle.y -= 5;
            if (keyState[SDL_SCANCODE_DOWN] && leftPaddle.y < height - PADDLE_HEIGHT) leftPaddle.y += 5;
        } else {
            if (leftPaddle.y + PADDLE_HEIGHT/2 > ball.y+20 && leftPaddle.y > 0) {
                leftPaddle.y -= 3;
            } else if (leftPaddle.y + PADDLE_HEIGHT/2 < ball.y-20 && leftPaddle.y < height - PADDLE_HEIGHT) {
                leftPaddle.y += 3;
            } else {
                // prevent stale back and forth along a horizontal line
                if (rand() % 2 == 0 && leftPaddle.y < height - PADDLE_HEIGHT) {
                    leftPaddle.y += 3;
                } else if (leftPaddle.y > 0) {
                    leftPaddle.y -= 3;
                }
            }
        }

        // check render.txt file to see if game should render
        if (framesSinceFileRead % 60 == 0) {
            FILE *renderFd = fopen("./render.txt", "r");
            if (renderFd == NULL) {
                printf("Error: Could not read render.txt\n");
                renderGame = 0;
            } else if (fscanf(renderFd, "%d", &renderGame) != 1) {
                printf("Error: Could not read render.txt\n");
                renderGame = 0;
            }
            fclose(renderFd);
            framesSinceFileRead = 0;
        } else {
            framesSinceFileRead++;
        }

        //
        // Graphics
        //

        if (renderGame) {
            // clear screen to black
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // black background
            SDL_RenderClear(renderer);

            // draw line and point text
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            SDL_RenderFillRectF(renderer, &line);

            // draw ball and paddles
            SDL_RenderFillRectF(renderer, &ball.rect);
            SDL_RenderFillRectF(renderer, &leftPaddle.rect);
            SDL_RenderFillRectF(renderer, &rightPaddle.rect);

            // display updated frame
            SDL_RenderPresent(renderer);

            SDL_Delay(1000/FPS);
        }
    }
    
    close(clientSock1);
    close(clientSock2);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

// sudo cp -r SDL2.framework /Library/Frameworks/
// gcc game.c -o game -Wall -F/Library/Frameworks -framework SDL2 -rpath /Library/Frameworks && ./game