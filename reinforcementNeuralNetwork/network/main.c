#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#include <neuralNetworkLib/neuralNetworkLib.h>

// By: Rowan Rothe

// declare the q network constants (same for both q network and target network)
#define Q_NETWORK_INPUTS 6 // network paddle y position, the current opponent paddle y position, the ball x and y position, and the ball x and y velocity. this comes to 6 inputs
#define Q_NETWORK_HIDDEN_LAYERS 2 // x hidden layers for this network
#define Q_NETWORK_HIDDEN_NODES (int[]){32, 16} // x hidden layers with x nodes in each layer
#define Q_NETWORK_OUTPUTS 3 // up, down, and stay still (index 0 is up, index 1 is down, index 2 is stay still)
#define Q_NETWORK_LEARNING_RATE 0.00001f // learning rate for gradient descent

// first index is the activation function for the hidden layers, the last index is the activation function for the output layer
#define ACTIVATIONS (int[]){1, 1, 3} // 0 for sigmoid, 1 for relu, 2 for tanh, 3 for linear, 4 for softmax

// optimizer stuff
#define OPTIMIZER_TYPE 3            // type: SGD (0), Momentum (1), RMSProp (2), Adam (3)
#define MOMENTUM_COEFFICIENT 0.9f   // number between 0 and 1. Zero to use no momentum
#define RMS_DECAY_RATE 0.999        // the decay rate of the gradients for RMSProp. How much of the previous gradients should we include in the new calculated gradient?

#define MINI_BATCH_SIZE 32//64

#define Q_UPDATE_FREQUENCY 1
#define TARGET_UPDATE_FREQUENCY 10000 // the frequency (in steps) at which the target network is updated with the q network

// this discount factor is the amount we will decrease every timestep's reward by. its a value between 0.9 and 0.99 and is applied to every value more and more as time goes on
// for example, if the rewards of 4 timesteps are 20, 3, -10, and 5, we apply the discount factor like this: the first reward is multiplied by the discount factor once (20 * discountFactor), then the second reward is multiplied by the discount factor twice (3 * discountFactor * discountFactor), then the third reward is multiplied by the discount factor three times (-10 * discountFactor * discountFactor * discountFactor), etc...
#define DISCOUNT_FACTOR 0.99f

// the greedy epsilon value is the amount of randomness the network will have in its actions
// the network will have a 1 - greedyEpsilon chance of choosing the action with the highest value and a greedyEpsilon chance of choosing a random action
// this value will start at GREEDY_EPSILON_START and decay to GREEDY_EPSILON_END by GREEDY_EPSILON_DECAY every episode
#define GREEDY_EPSILON_START 1.0f
#define GREEDY_EPSILON_END 0.1f
#define GREEDY_EPSILON_DECAY 0.9996f//0.996f//0.9996f //0.9995f or 0.996f or 0.999f or 0.99f

// replay buffer size
#define REPLAY_BUFFER_SIZE 1000000 // 50000 // 100000

// the amount of steps the agent will take before starting to train
#define START_TRAINING 10000 //50000 or 10000

#define EPISODES 5000//10000 // the number of episodes to be trained on

#define MAX_MOVES 5000 // the maximum number of moves the agent can make before the episode/game ends

#define SOCKET_PORT 1234
#define BUFFER_SIZE 1024

// game constants
#define WIDTH 800
#define HEIGHT 600

// Game state structure
typedef struct {
    float netPaddleY;
    float oppPaddleY;
    int networkPoints;
    int opponentPoints;
    float ballX;
    float ballY;
    float ballVelX;
    float ballVelY;
} GameState;


// declare transition struct
typedef struct {
    float *state;       // Pointer to the state array
    float action;       // The action taken
    float reward;       // The reward received
    int done;           // Whether the episode terminated (1 for true, 0 for false)
} Transition;

// declare replay buffer struct
typedef struct {
    Transition *buffer; // Array of transitions (State, Action, Reward, Done)
    int capacity;       // Maximum number of transitions
    int size;           // Current number of stored transitions
    int index;          // Current index for overwriting transitions
    int stateSize;      // Size of the state array
} ReplayBuffer;

// Function to format the move as a string
void formatMove(char *buffer, float move) {
    snprintf(buffer, BUFFER_SIZE, "{%f}", move);
}

// Function to format the game constants as a string
void formatGameConstants(char *buffer, int width, int height) {
    snprintf(buffer, BUFFER_SIZE, "{%d, %d}", width, height);
}

// implement the replay buffer based on REPLAY_BUFFER_SIZE and delayed training based on START_TRAINING

// start with randomly choosing states, actions, rewards, and next states and train it on that
// the target value is the reward + gamma * the estimated value of the action chosen in a certain state by the online network

// consider batch updates for the networks and update the target network on a move basis rather than a per episode basis
// also make the epsilon decay linearly over timesteps rather than by a certian value every step. (e.g. delay linearly from 500,000 to 1,000,000 timesteps instead of decay 0.01 every step or episode)
// IMPORTANT: mini-batch Stochastic Gradient Descent. Implement with mnist first then move to here. might have to restructure how the library works as to allow for different gradient algos

ReplayBuffer *initializeBuffer(int capacity, int stateSize) {
    ReplayBuffer *replayBuffer = malloc(sizeof(ReplayBuffer));
    replayBuffer->buffer = malloc(sizeof(Transition) * capacity);
    replayBuffer->capacity = capacity;
    replayBuffer->size = 0;
    replayBuffer->index = 0;
    replayBuffer->stateSize = stateSize;

    // Initialize state array for each transition
    for (int i = 0; i < capacity; i++) {
        replayBuffer->buffer[i].state = malloc(sizeof(float) * stateSize);
    }

    return replayBuffer;
}

void freeBuffer(ReplayBuffer *replayBuffer) {
    for (int i = 0; i < replayBuffer->capacity; i++) {
        free(replayBuffer->buffer[i].state);
    }
    free(replayBuffer->buffer);
    free(replayBuffer);
}

void addTransition(ReplayBuffer *replayBuffer, float *state, float action, float reward, int done) {
    Transition *transition = &replayBuffer->buffer[replayBuffer->index];
    
    // Copy the state and next state
    memcpy(transition->state, state, sizeof(float) * replayBuffer->stateSize);
    
    // Set other values
    transition->action = action;
    transition->reward = reward;
    transition->done = done;

    // Update the index and size
    replayBuffer->index = (replayBuffer->index + 1) % replayBuffer->capacity;
    if (replayBuffer->size < replayBuffer->capacity) {
        replayBuffer->size++;
    }
}

int floatArgmax(float *arr, int size) {
    int maxIndex = 0;
    for (int i = 0; i < size; i++) {
        if (arr[i] > arr[maxIndex]) {
            maxIndex = i;
        }
    }

    return maxIndex;
}


int main() {
    // declare network variables for the q network and target network
    Network *qNetwork;
    Network *targetNetwork;

    // seed random number generator
    printf("Seeding random number generator\n");
    srand(time(NULL));


    // clear writing discounted reward file & open for appending
    FILE *discountedRewardFile = fopen("discountedReward.txt", "w");
    if (discountedRewardFile == NULL) {
        perror("error opening discountedReward.txt");
        return 1;
    }

    discountedRewardFile = fopen("discountedReward.txt", "a");
    if (discountedRewardFile == NULL) {
        perror("error opening discountedReward.txt");
        return 1;
    }


    // ask user if they would like to load the networks
    char load;
    printf("Would you like to load the network? (y/n): ");
    scanf("%c", &load);
    if (load == 'y') {
        printf("Loading Q network...\n");
        qNetwork = importNetworkJSON("qNetwork.json");
        if (qNetwork == NULL) {
            perror("Error loading Q network");
            return 1;
        }

        printf("Loading target network...\n");
        targetNetwork = importNetworkJSON("targetNetwork.json");
        if (targetNetwork == NULL) {
            perror("Error loading target network");
            return 1;
        }

        printf("Networks loaded\n");
    } else if (load != 'n') {
        printf("Invalid input\n");
        return 1;
    }


    // create Q and target networks
    if (load == 'n') {
        printf("Creating Q network\n");
        qNetwork = createNetwork(Q_NETWORK_INPUTS, Q_NETWORK_HIDDEN_LAYERS, Q_NETWORK_HIDDEN_NODES, Q_NETWORK_OUTPUTS, ACTIVATIONS);
        if (qNetwork == NULL) {
            perror("Error creating Q network");
            return 1;
        }

        printf("Creating target network\n");
        targetNetwork = createNetwork(Q_NETWORK_INPUTS, Q_NETWORK_HIDDEN_LAYERS, Q_NETWORK_HIDDEN_NODES, Q_NETWORK_OUTPUTS, ACTIVATIONS);
        if (targetNetwork == NULL) {
            perror("Error creating target network");
            return 1;
        }

        // initialize optimizer for q network and target networks
        printf("Initializing Optimizer for Q network\n");
        Optimizer *optimizer = initializeOptimizer(qNetwork, OPTIMIZER_TYPE, Q_NETWORK_LEARNING_RATE);
        if (optimizer == NULL) {
            perror("Error initializing the optimizer");
            return 1;
        }

        // set optimizer parameters
        optimizer->momentumCoefficient = MOMENTUM_COEFFICIENT;
        optimizer->RMSPropDecay = RMS_DECAY_RATE;

        // apply optimizer to the network
        applyOptimizer(qNetwork, optimizer);


        printf("Initializing Optimizer for target network\n");
        optimizer = initializeOptimizer(targetNetwork, OPTIMIZER_TYPE, Q_NETWORK_LEARNING_RATE);
        if (optimizer == NULL) {
            perror("Error initializing the optimizer");
            return 1;
        }

        // set optimizer parameters
        optimizer->momentumCoefficient = MOMENTUM_COEFFICIENT;
        optimizer->RMSPropDecay = RMS_DECAY_RATE;

        // apply optimizer to the network
        applyOptimizer(targetNetwork, optimizer);
    }


    // set up sockets for data communication between network program and game
    printf("Initializing socket communication\n");

    int sock, clientSock;
    struct sockaddr_in server, client;
    char buffer[BUFFER_SIZE];
    socklen_t clientLen = sizeof(client);

    // initialize socket and server structure
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1) {
        perror("Could not create socket\n");
        return 1;
    }

    server.sin_family = AF_INET;
    server.sin_addr.s_addr = inet_addr("127.0.0.1");
    server.sin_port = htons(SOCKET_PORT);

    // allow server to reuse port
    int opt = 1;
    if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt failed");
        return 1;
    }

    // bind to port & addr
    if (bind(sock, (struct sockaddr *)&server, sizeof(server)) < 0) {
        perror("Bind failed");
        return 1;
    }

    listen(sock, 1);

    printf("Server listening on port %d (please connect the python game in another window (python3 game.py))\n", SOCKET_PORT);

    clientSock = accept(sock, (struct sockaddr *)&client, &clientLen);
    if (clientSock < 0) {
        perror("Accept failed");
        return 1;
    }

    printf("Client connected\n");

    // send game constants to client
    // format: {width,height}
    printf("Sending game constants\n");
    int width = WIDTH;
    int height = HEIGHT;
    formatGameConstants(buffer, WIDTH, HEIGHT);
    if (send(clientSock, buffer, strlen(buffer), 0) < 0) {
        perror("Failed to send game constants");
        return 1;
    }
    printf("Sent game constants: %s\n", buffer);

    if (load == 'y') {
        printf("Would you like to continue training the network (otherwise the program will begin playing the game)? (y/n): ");
        while ((load = getchar()) != '\n'); // consume the newline character
        char continueTraining;
        scanf("%c", &continueTraining);
        if (continueTraining == 'y') {
            // do nothing and pass through to training
            printf("Continuing training...\n");
        } else if (continueTraining == 'n') {
            // test network with user input
            goto playGame;
        } else {
            printf("Invalid input\n");
            return 1;
        }
    }



    // train the network by playing the game
    printf("Training network...\n\n");


    // declare variables for the game

    // this array keeps track of the State, Action, and Reward of a certian timestep.
    ReplayBuffer *replayBuffer = initializeBuffer(REPLAY_BUFFER_SIZE, Q_NETWORK_INPUTS);

    // game points variables
    int networkPoints = 0;
    int opponentPoints = 0;
    int prevNetworkPoints = 0;
    int prevOpponentPoints = 0;

    // arrays to store the current state and previous state (network paddle y position, the opponent paddle y position, the ball x and y position, and the ball x and y velocity)
    // 0: network paddle y, 1: opponent paddle y, 2: ball x, 3: ball y, 4: ball x velocity, 5: ball y velocity
    float state[Q_NETWORK_INPUTS] = {0};
    float prevState[Q_NETWORK_INPUTS] = {0};

    // pointers to state & prevState arrays to more easily understand reward functions
    // state
    float *netPaddleY = &state[0];
    float *oppPaddleY = &state[1];
    float *ballX = &state[2];
    float *ballY = &state[3];
    float *ballVelX = &state[4];
    float *ballVelY = &state[5];

    // prevState
    float *prevNetPaddleY = &prevState[0];
    float *prevOppPaddleY = &prevState[1];
    float *prevBallX = &prevState[2];
    float *prevBallY = &prevState[3];
    float *prevBallVelX = &prevState[4];
    float *prevBallVelY = &prevState[5];

    // current and previous move/action that the network has made
    // initialize randomly to 1 or 0
    float move;
    float prevMove;

    // current and previous reward that the network got in the current state
    float reward = 0.0f;
    float prevReward = 0.0f;

    // flag to signal the end of the game/episode
    // 0 :game is still going, 1: the game has ended
    int endGame = 0;

    // keeps track of how many steps have been taken since the last Q network update and variable for target network equivalent
    int stepCountSinceQUpdate = 0;
    int stepCountSinceTargetUpdate = 0;

    // keeps track of how many timesteps have passed since the beginning of the training (for delayed training)
    int stepsSinceBeginning = 0;

    // keeps track of how many steps were taken within an episode
    int episodeMoveCount = 0;

    // this variable is used to keep track of the greedy epsilon value
    float greedyEpsilon = GREEDY_EPSILON_START;

    // for each episode, the network will play the game until a point is scored
    // after a point is scored/an episode has ended, the network will be rewarded or punished based on the moves it made
    for (int episode = 0; episode < EPISODES; episode++) {
        // repeat until a point is scored or the network has made the maximum amount of moves
        for (int moveCount = 0; moveCount < MAX_MOVES; moveCount++) {
            // set current game state from state files (ensuring the data is new first)
            do {
                // recieve game state
                if (recv(clientSock, buffer, sizeof(buffer) - 1, 0) < 0) {
                    perror("Failed to recieve game state");
                }

                // parse recieved game state
                // the format is as follows: {networkPaddleY,opponentPaddleY,networkPoints,opponentPoints,ballX,ballY,ballVelocityX,ballVelocityY}
                sscanf(buffer, "{%f,%f,%d,%d,%f,%f,%f,%f}\n", &state[0], &state[1], &networkPoints, &opponentPoints, &state[2], &state[3], &state[4], &state[5]);

                // normalize the state between 0 and 1 using game constants like width and height.
                // in the case of ball velocity, it stays between -5 and 5, so normalize it between -1 and 1
                state[0] = (float)state[0] / (float)height; // network paddle y
                state[1] = (float)state[1] / (float)height; // opponent paddle y
                state[2] = (float)state[2] / (float)width;  // ball x
                state[3] = (float)state[3] / (float)height; // ball y
                state[4] = (float)state[4] / 5.0f;          // ball velocity x
                state[5] = (float)state[5] / 5.0f;          // ball velocity y
            } while (state[0] == prevState[0] && state[1] == prevState[1] && networkPoints == prevNetworkPoints && opponentPoints == prevOpponentPoints && state[2] == prevState[2] && state[3] == prevState[3] && state[4] == prevState[4] && state[5] == prevState[5]);

            if (moveCount == 0) printf("Game #%d started\n", episode + 1);

            // set the reward here to the last move because we are trying to see what the reward was for the move made in the last timestep and we cant do that until we get an updated game state from the environment
            // basically, we just got reward feedback from the env so match the reward of that state up properly with the corresponding reward
            
            if (moveCount > 0) {
                // reward the network if it is on top of the ball or moves in the direction of the ball. otherwise punish
                if (*prevBallY >= *prevNetPaddleY && *prevBallY*height <= *prevNetPaddleY*height+100) {
                    // reward the network for being on top of the ball
                    // if (ballY >= networkPaddleY || ballY <= networkPaddleY+100)
                    prevReward += 0.015f;
                } else if ((*prevNetPaddleY > *prevBallY && prevMove == 1.0f) || (*prevNetPaddleY + (100/height) < *prevBallY && prevMove == 0.0f)) {
                    //if ((networkPaddleY > ballY && move == up) || ((networkPaddleY+100) < ballY && move == down))
                    prevReward += 0.01f;
                } else {
                    prevReward += -0.01f;//-0.005f;
                }

                // reward the network for hitting the ball back / touching the ball
                if (*ballVelX < *prevBallVelX && *ballVelX < 0 && *ballVelX * width > width / 2) {
                    // if (ballVelocityX < prevBallVelocityX) and (ballVelocityX < 0) and (ballX > width / 2)
                    prevReward += 0.5f;
                }

                // penalize the network for hitting the ball in a horizontal line
                /*if (&prevBallVelY == 0 && *prevBallVelX < 0) {
                    // if (ballVelocityY == 0 && ballVelocityX < 0)
                    prevReward += -0.015;
                }*/

                // punish the network for trying to move outside of the game bounds
                if ((*prevNetPaddleY == 0 && prevMove == 1.0f) || (*prevNetPaddleY * height == height-100 && prevMove == 0.0f)) {
                    // if ((networkPaddleY == 0 && prevMove == up) || (networkPaddleY == height && prevMove == down)) {
                    //prevReward += -0.2f;
                }

                // reward the network if the network has scored a point
                // punish it if the opponent has scored a point
                if (networkPoints > 0 || opponentPoints > 0) {
                    if (networkPoints != prevNetworkPoints || opponentPoints != prevOpponentPoints) {
                        if (networkPoints > prevNetworkPoints) {
                            reward += 1.0f;
                        } else if (opponentPoints > prevOpponentPoints) {
                            reward += -1.0f;
                        }
                    }
                }

                // reward the network proportional to the distance between the ball and the paddle
                // the closer the ball is to the center of the paddle, the more the network is rewarded (positive)
                // the further the ball is from the center of the paddle, the more the network is punished (negative)
                // the range of the punishments and rewards are +- 0.15 points
                /*if (ballX > width / 2 && ballVelocityX > 0) {
                    rewards[episodeMoveCount-1] += (0.15f - ((float)abs(ballY - networkPaddleY+50) / height)) * 0.3f;
                }*/

                // reward the network if the ball is close to the paddle
                /*if (ballY - networkPaddleY > 0 && ballY - networkPaddleY < 100 && ballVelocityX > 0 && ballX > width / 2) {
                    rewards[episodeMoveCount-1] += 5.0f;
                    //printf("rewarded\n");
                } else if ((ballY - networkPaddleY < 0 || ballY - networkPaddleY > 100) && ballVelocityX > 0 && ballX > width / 2) {
                    //rewards[episodeMoveCount-1] += -0.1f;
                    //rewards[episodeMoveCount-1] += -(fabsf(ballY - networkPaddleY) / height) * 0.2f;
                    //printf("punished\n");
                }*/
            }

            // feed the current game state through the q network
            feedForward(qNetwork, state);


            // generate random number for greedy epsilon
            // if the random number is less than the greedy epsilon, then treat it as a random move
            // otherwise, treat it as the network's decision
            if ((float)rand() / RAND_MAX <= greedyEpsilon) {
                float randNum = (float)rand() / RAND_MAX;
                if (randNum <= 0.3f) {
                    move = 0.0f;
                } else if (randNum >= 0.6f) {
                    move = 1.0f;
                } else {
                    move = 0.5f;
                }
            } else {
                // get the actor network's decision and store it in the decisions array
                // the selected move is the highest value output (argmax of the output layer)
                // index 0 is up, index 1 is down, index 2 is stay still
                int maxIndex = floatArgmax(qNetwork->layers[qNetwork->layerCount - 1].values, Q_NETWORK_OUTPUTS);
                if (maxIndex == 0) {
                    move = 1.0f;
                } else if (maxIndex == 1) {
                    move = 0.0f;
                } else {
                    move = 0.5f;
                }
            }

            // append states to the replay buffer but delayed because we wont have the next state until the loop after
            if (moveCount > 0) {
                addTransition(replayBuffer, prevState, prevMove, prevReward, endGame); // end game would always be zero here. this is useful only when we update the buffer right before the game ends which i do above
            }

            // set endGame flag to 1 if a point has been scored only if points are greater than 0
            // or if reached MAX_MOVES
            // specifically here so it doesnt interfere with the addTransition above
            if (networkPoints != prevNetworkPoints || opponentPoints != prevOpponentPoints || moveCount == MAX_MOVES-1) {
                // set end game flag
                endGame = 1;
            }

            // send move to python program with the format {move}
            // format: {1.000000} for up, {0.000000} for down, {0.500000} for stay still
            formatMove(buffer, move);
            if (send(clientSock, buffer, strlen(buffer), 0) < 0) {
                perror("Failed to send move");
            }

            // check if the game has ended
            if (endGame) {
                // update replay buffer
                addTransition(replayBuffer, state, move, reward, endGame);

                // set previous variables
                // set prevState
                for (int i = 0; i < Q_NETWORK_INPUTS; i++) {
                    prevState[i] = state[i];
                }
                prevMove = move;
                prevReward = reward;
                move = 0;
                reward = 0;

                // update step count vars
                if (stepsSinceBeginning >= START_TRAINING) {
                    stepCountSinceQUpdate++;
                    stepCountSinceTargetUpdate++;
                }

                // update episode move count
                episodeMoveCount++;

                // update steps since beginning for delayed training
                if (stepsSinceBeginning < START_TRAINING) stepsSinceBeginning++;

                // end the game and reset the endGame flag
                endGame = 0;
                break;
            }

            // set previous variables
            // set prevState
            for (int i = 0; i < Q_NETWORK_INPUTS; i++) {
                prevState[i] = state[i];
            }
            prevMove = move;
            prevReward = reward;

            // clear current variables
            move = 0;
            reward = 0;

            // update step count vars
            if (stepsSinceBeginning >= START_TRAINING) {
                stepCountSinceQUpdate++;
                stepCountSinceTargetUpdate++;
            }

            // update episode move count
            episodeMoveCount++;

            // update steps since beginning for delayed training
            if (stepsSinceBeginning < START_TRAINING) stepsSinceBeginning++;


            // update Q network every Q_UPDATE_FREQUENCY steps (only once training begins)
            if (stepCountSinceQUpdate >= Q_UPDATE_FREQUENCY && stepsSinceBeginning >= START_TRAINING) {
                float targetQValue = 0.0f;                      // the target q value for the state
                int targetQValueIndex = 0;                      // the index of the target q value
                float backPropagatedValues[Q_NETWORK_OUTPUTS];  // the backpropagated values for the output layer

                // process the batches (calculate gradients over the whole batch)
                for (int i = 0; i < MINI_BATCH_SIZE; i++) {
                    // get random index for batch
                    int randBufferIndex = rand() % (replayBuffer->size-1);

                    // calculate the q value for the current state
                    feedForward(qNetwork, replayBuffer->buffer[randBufferIndex].state);

                    // calculate the target q value for the next state
                    feedForward(targetNetwork, replayBuffer->buffer[randBufferIndex+1].state);

                    if (replayBuffer->buffer[randBufferIndex].action == 1.0f) {
                        targetQValue = targetNetwork->layers[targetNetwork->layerCount - 1].values[0];
                        targetQValueIndex = 0;
                    } else if (replayBuffer->buffer[randBufferIndex].action == 0.0f) {
                        targetQValue = targetNetwork->layers[targetNetwork->layerCount - 1].values[1];
                        targetQValueIndex = 1;
                    } else if (replayBuffer->buffer[randBufferIndex].action == 0.5f) {
                        targetQValue = targetNetwork->layers[targetNetwork->layerCount - 1].values[2];
                        targetQValueIndex = 2;
                    } else {
                        printf("Unexpected move value\n");
                        continue;
                    }

                    if (replayBuffer->buffer[randBufferIndex].done == 1) {
                        // special case if terminal state
                        targetQValue = replayBuffer->buffer[randBufferIndex].reward;
                    } else {
                        // apply reward/punishment to the target q value to get the full updated target
                        targetQValue = replayBuffer->buffer[randBufferIndex].reward + DISCOUNT_FACTOR * targetQValue;
                    }

                    // setup backpropagated value array to be used in backpropagation
                    for (int i = 0; i < Q_NETWORK_OUTPUTS; i++) {
                        if (i == targetQValueIndex) {
                            backPropagatedValues[i] = targetQValue;
                        } else {
                            backPropagatedValues[i] = qNetwork->layers[qNetwork->layerCount - 1].values[i];
                        }
                    }

                    computeGradients(qNetwork, backPropagatedValues);
                }

                // average the accumulated batch gradients, update parameters, & zero the gradients to prepare for next batch
                averageAccumulatedGradients(qNetwork, MINI_BATCH_SIZE);
                SGDUpdate(qNetwork);
                zeroGradients(qNetwork);

                stepCountSinceQUpdate = 0;
            }

            // update the target network every TARGET_UPDATE_FREQUENCY episodes (only once training begins)
            if (stepCountSinceTargetUpdate == TARGET_UPDATE_FREQUENCY && stepsSinceBeginning >= START_TRAINING) {
                // copy the q network to the target network
                copyNetwork(targetNetwork, qNetwork);
                stepCountSinceTargetUpdate = 0;
            }
        }

        printf("Game #%d completed\n", episode + 1);

        // calculate the total discounted reward for the episode and print
        // calculate the total discounted reward for the episode and print
        float totalDiscountedReward = 0.0f;
        int moveIndex = replayBuffer->index - episodeMoveCount;
        if (moveIndex < 0) {
            moveIndex = replayBuffer->capacity + moveIndex; // if i subtract a negative, it adds instead
        }
        for (int i = 0; i < episodeMoveCount; i++) {
            totalDiscountedReward = replayBuffer->buffer[moveIndex].reward + DISCOUNT_FACTOR * totalDiscountedReward;
            moveIndex = (moveIndex + 1) % replayBuffer->capacity;  // Safe increment with modulo
        }
        printf("Total discounted reward: %f\n", totalDiscountedReward);
        printf("Timesteps taken for episode: %d\n\n", episodeMoveCount);

        // write discounted reward to file for graphing
        if (episode % 10 == 0) { 
            fprintf(discountedRewardFile, "%d, %f\n", episode, totalDiscountedReward);
        }

        // update prev points
        prevNetworkPoints = networkPoints;
        prevOpponentPoints = opponentPoints;

        // reset the move count for the current episode
        episodeMoveCount = 0;

        // update the greedy epsilon value only if training has started
        if (greedyEpsilon > GREEDY_EPSILON_END && stepsSinceBeginning >= START_TRAINING) {
            greedyEpsilon *= GREEDY_EPSILON_DECAY;
        }
    }

    // finished training
    printf("Finished training\n");
    // play cool sound :D
    system("afplay /System/Library/Sounds/Blow.aiff ");


    // ask user if they would like to save the network
    char save = '\0';

    // consume the newline character
    while ((save = getchar()) != '\n');

    // ask user if they would like to save the network
    do {
        printf("Would you like to save the network? (y/n): ");
        char save;
        scanf("%c", &save);
        if (save == 'y') {
            printf("Saving Q network...\n");
            exportNetworkJSON(qNetwork, "qNetwork.json");
            printf("Q network saved\n");

            printf("Saving target network...\n");
            exportNetworkJSON(targetNetwork, "targetNetwork.json");
            printf("Target network saved\n");

            // free networks, decisions, rewards, and game states
            freeNetwork(qNetwork);
            freeNetwork(targetNetwork);
            freeBuffer(replayBuffer);
            
            return 0;
        } else if (save == 'n') {
            printf("Network not saved\n");
            
            // free networks, decisions, rewards, and game states
            freeNetwork(qNetwork);
            freeNetwork(targetNetwork);
            freeBuffer(replayBuffer);

            return 0;
        } else {
            printf("Invalid input\n");
        }
    } while (save != 'y' && save != 'n');


    // run the network against the opponent
    playGame:

    // clear relevant game variables
    printf("Clearing game variables\n");
    networkPoints = 0;
    opponentPoints = 0;
    prevNetworkPoints = 0;
    prevOpponentPoints = 0;
    for (int i = 0; i < Q_NETWORK_INPUTS; i++) {
        state[i] = 0;
        prevState[i] = 0;
    }

    printf("Running network against opponent (please start the python game in another window (python3 game.py))\n");

    // run game loop indefinitely
    while (1) {
        // set current game state from state files (ensuring the data is new first)
        do {
            // recieve game state
            if (recv(clientSock, buffer, sizeof(buffer) - 1, 0) < 0) {
                perror("Failed to recieve game state");
            }

            // parse recieved game state
            // the format is as follows: {networkPaddleY,opponentPaddleY,networkPoints,opponentPoints,ballX,ballY,ballVelocityX,ballVelocityY}
            sscanf(buffer, "{%f,%f,%d,%d,%f,%f,%f,%f}\n", &state[0], &state[1], &networkPoints, &opponentPoints, &state[2], &state[3], &state[4], &state[5]);

            // normalize the state between 0 and 1 using game constants like width and height.
            // in the case of ball velocity, it stays between -5 and 5, so normalize it between -1 and 1
            state[0] = (float)state[0] / (float)height; // network paddle y
            state[1] = (float)state[1] / (float)height; // opponent paddle y
            state[2] = (float)state[2] / (float)width;  // ball x
            state[3] = (float)state[3] / (float)height; // ball y
            state[4] = (float)state[4] / 5.0f;          // ball velocity x
            state[5] = (float)state[5] / 5.0f;          // ball velocity y
        } while (state[0] == prevState[0] && state[1] == prevState[1] && networkPoints == prevNetworkPoints && opponentPoints == prevOpponentPoints && state[2] == prevState[2] && state[3] == prevState[3] && state[4] == prevState[4] && state[5] == prevState[5]);

        // feed the input through the network
        feedForward(qNetwork, state);

        // get the actor network's decision and store it in the decisions array
        // the selected move is the highest value output (argmax of the output layer)
        // index 0 is up, index 1 is down, index 2 is stay still
        int maxIndex = floatArgmax(qNetwork->layers[qNetwork->layerCount - 1].values, Q_NETWORK_OUTPUTS);
        if (maxIndex == 0) {
            move = 1.0f;
        } else if (maxIndex == 1) {
            move = 0.0f;
        } else {
            move = 0.5f;
        }

        // send move to python program with the format {move}
        // format: {1.000000} for up, {0.000000} for down, {0.500000} for stay still
        formatMove(buffer, move);
        if (send(clientSock, buffer, strlen(buffer), 0) < 0) {
            perror("Failed to send move");
        }
    }

    // free networks, decisions, rewards, and game states
    freeNetwork(qNetwork);
    freeNetwork(targetNetwork);
    freeBuffer(replayBuffer);

    // close the socket fd
    close(clientSock);
    close(sock);

    return 0;
}

// clear && clear && gcc main.c -L/usr/local/lib -lneuralNetworkLib -I/usr/local/include -o main && ./main