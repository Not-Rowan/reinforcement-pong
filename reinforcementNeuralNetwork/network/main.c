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
#define Q_NETWORK_HIDDEN_NODES (int[]){64, 64} // x hidden layers with x nodes in each layer
#define Q_NETWORK_OUTPUTS 3 // up, down, and stay still (index 0 is up, index 1 is down, index 2 is stay still)
#define Q_NETWORK_LEARNING_RATE 1e-5 // learning rate for gradient descent

// first index is the activation function for the hidden layers, the last index is the activation function for the output layer
#define ACTIVATIONS (int[]){1, 1, 3} // 0 for sigmoid, 1 for relu, 2 for tanh, 3 for linear, 4 for softmax

// optimizer stuff
#define OPTIMIZER_TYPE 3            // type: SGD (0), Momentum (1), RMSProp (2), Adam (3)
#define MOMENTUM_COEFFICIENT 0.9f   // number between 0 and 1. Zero to use no momentum
#define RMS_DECAY_RATE 0.999        // the decay rate of the gradients for RMSProp. How much of the previous gradients should we include in the new calculated gradient?

#define MINI_BATCH_SIZE 32

#define Q_UPDATE_FREQUENCY 1
#define TARGET_UPDATE_FREQUENCY 10000 // the frequency (in steps) at which the target network is updated with the q network

// this discount factor is the amount we will decrease every timestep's reward by. its a value between 0.9 and 0.99 and is applied to every value more and more as time goes on
// for example, if the rewards of 4 timesteps are 20, 3, -10, and 5, we apply the discount factor like this: the first reward is multiplied by the discount factor once (20 * discountFactor), then the second reward is multiplied by the discount factor twice (3 * discountFactor * discountFactor), then the third reward is multiplied by the discount factor three times (-10 * discountFactor * discountFactor * discountFactor), etc...
#define DISCOUNT_FACTOR 0.99f

// the greedy epsilon value is the amount of randomness the network will have in its actions
// the network will have a 1 - greedyEpsilon chance of choosing the action with the highest value and a greedyEpsilon chance of choosing a random action
// this value will start at GREEDY_EPSILON_START and decay to GREEDY_EPSILON_END over GREEDY_EPSILON_DECAY_STEPS steps
#define GREEDY_EPSILON_START 1.0f
#define GREEDY_EPSILON_END 0.05f
#define GREEDY_EPSILON_DECAY_STEPS 2000000

// replay buffer size
#define REPLAY_BUFFER_SIZE 100000

// the amount of steps the agent will take before starting to train
#define START_TRAINING 0//10000

#define EPISODES 10000//5000 // the number of episodes to be trained on

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

    float reward;
    float move;
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

void translateState(float *stateArr, GameState *state) {
    stateArr[0] = state->netPaddleY;
    stateArr[1] = state->oppPaddleY;
    stateArr[2] = state->ballX;
    stateArr[3] = state->ballY;
    stateArr[4] = state->ballVelX;
    stateArr[5] = state->ballVelY;
}

// Function to format the move as a string
void formatMove(char *buffer, float move) {
    snprintf(buffer, BUFFER_SIZE, "{%f}", move);
}

// Function to format the game constants as a string
void formatGameConstants(char *buffer, int width, int height) {
    snprintf(buffer, BUFFER_SIZE, "{%d, %d}", width, height);
}



// Function to normalize any float to a range of -1 to 1
float normalizeFloat(float value, float minValue, float maxValue) {
    if (minValue == maxValue) return 0.0; // prevent division by zero
    return 2 * ((value-minValue) / (maxValue-minValue)) - 1;
}

// Function to un-normalize any float from a range of -1 to 1 back to the original range
float unnormalizeFloat(float normalizedValue, float minValue, float maxValue) {
    return (normalizedValue + 1) / 2 * (maxValue - minValue) + minValue;
}



// start with randomly choosing states, actions, rewards, and next states and train it on that
// the target value is the reward + gamma * the estimated value of the action chosen in a certain state by the online network
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
        replayBuffer->buffer[i].state = NULL;
    }
    
    free(replayBuffer->buffer);
    replayBuffer->buffer = NULL;

    free(replayBuffer);
    replayBuffer = NULL;
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
        if (arr[i] > arr[maxIndex] + 1e-8f) {
            maxIndex = i;
        }
    }

    return maxIndex;
}

typedef struct NetworkInfo {
    int serverSock, clientSock;
    struct sockaddr_in server, client;
    socklen_t clientLen;
} SocketInfo;

SocketInfo *connectToClient() {
    SocketInfo *sockInfo = malloc(sizeof(SocketInfo));
    if (!sockInfo) { perror("malloc failed"); return NULL; }

    socklen_t clientLen = sizeof(sockInfo->client);

    // initialize socket and server structure
    sockInfo->serverSock = socket(AF_INET, SOCK_STREAM, 0);
    if (sockInfo->serverSock == -1) {
        perror("Could not create socket\n");
        free(sockInfo);
        return NULL;
    }

    sockInfo->server.sin_family = AF_INET;
    sockInfo->server.sin_addr.s_addr = inet_addr("127.0.0.1");
    sockInfo->server.sin_port = htons(SOCKET_PORT);

    // allow server to reuse port
    int opt = 1;
    if (setsockopt(sockInfo->serverSock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt failed");
        free(sockInfo);
        return NULL;
    }

    // bind to port & addr
    if (bind(sockInfo->serverSock, (struct sockaddr *)&sockInfo->server, sizeof(sockInfo->server)) < 0) {
        perror("Bind failed");
        free(sockInfo);
        return NULL;
    }

    listen(sockInfo->serverSock, 1);

    printf("Server listening on port %d (please connect the python game in another window (python3 game.py))\n", SOCKET_PORT);

    sockInfo->clientSock = accept(sockInfo->serverSock, (struct sockaddr *)&sockInfo->client, &clientLen);
    if (sockInfo->clientSock < 0) {
        perror("Accept failed");
        free(sockInfo);
        return NULL;
    }

    return sockInfo;
}

void disconnectFromClient(SocketInfo *sockInfo) {
    if (sockInfo == NULL) {
        perror("trying to disconnect from uninitialized client");
        return;
    }

    close(sockInfo->serverSock);
    close(sockInfo->clientSock);
    free(sockInfo);
}


void calculateReward(GameState *state, GameState *nextState, int stepsTaken) {
    // reward the network if it is on top of the ball or moves in the direction of the ball. otherwise punish
    if (state->ballY >= state->netPaddleY && unnormalizeFloat(state->ballY, 0.0f, (float)HEIGHT) <= unnormalizeFloat(state->netPaddleY, 0.0f, (float)HEIGHT)+100) {
        // reward the network for being on top of the ball
        // if (ballY >= networkPaddleY || ballY <= networkPaddleY+100)
        //state->reward += 0.015f;
    } else if ((state->netPaddleY > state->ballY && state->move == 1.0f) || (unnormalizeFloat(state->netPaddleY, 0.0f, HEIGHT)+100 < state->ballY && state->move == 0.0f)) {
        //if ((networkPaddleY > ballY && move == up) || ((networkPaddleY+100) < ballY && move == down))
        //state->reward += 0.01f;
    } else {
        //state->reward += -0.1f;//-0.005f;
    }

    // reward the network for hitting the ball back / touching the ball
    if (nextState->ballVelX < state->ballVelX && nextState->ballVelX < 0 && unnormalizeFloat(nextState->ballX, 0.0f, WIDTH) > WIDTH / 2) {
        // if (ballVelocityX < prevBallVelocityX) and (ballVelocityX < 0) and (ballX > width / 2)
        //state->reward += 1.0f;
    }

    // penalize the network for hitting the ball in a horizontal line
    /*if (state->ballVelY == 0 && state->ballVelX < 0) {
        // if (ballVelocityY == 0 && ballVelocityX < 0)
        state->reward += -0.015;
    }*/

    // punish the network for trying to move outside of the game bounds
    if ((state->netPaddleY == 0 && state->move == 1.0f) || (unnormalizeFloat(state->netPaddleY, 0.0f, HEIGHT) == HEIGHT-100 && state->move == 0.0f)) {
        // if ((networkPaddleY == 0 && prevMove == up) || (networkPaddleY == height && prevMove == down)) {
        //state->reward += -0.2f;
    }

    // reward the network if the network has scored a point
    // punish it if the opponent has scored a point
    if (nextState->networkPoints > 0 || nextState->opponentPoints > 0) {
        if (nextState->networkPoints != state->networkPoints || nextState->opponentPoints != state->opponentPoints) {
            if (nextState->networkPoints > state->networkPoints) {
                //nextState->reward += 3.0f * (1.0f - ((float)stepsTaken / (float)MAX_MOVES)); // less steps = higher reward
                nextState->reward += 3.0f;
            } else if (nextState->opponentPoints > state->opponentPoints) {
                //nextState->reward += -3.0f * ((float)stepsTaken/(float)MAX_MOVES); // less steps = less punishment
                nextState->reward += -3.0f;
            }
        }
    }

    // punishment for taking long
    //nextState->reward -= 0.01f;
}


int main() {
    // set method selection to use BLAS instead of BASIC
    setLinAlgSelectedMethod(BLAS);

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
            freeNetwork(qNetwork);
            return 1;
        }

        printf("Networks loaded\n");

        // continue training?
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
            freeNetwork(qNetwork);
            freeNetwork(targetNetwork);
            return 1;
        }
    } else if (load == 'n') {
        // create Q and target networks
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
            freeNetwork(qNetwork);
            return 1;
        }

        // initialize optimizer for q network and target networks
        printf("Initializing Optimizer for Q network\n");
        Optimizer *optimizer = initializeOptimizer(qNetwork, OPTIMIZER_TYPE, Q_NETWORK_LEARNING_RATE);
        if (optimizer == NULL) {
            perror("Error initializing the optimizer");
            freeNetwork(qNetwork);
            freeNetwork(targetNetwork);
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
            freeNetwork(qNetwork);
            freeNetwork(targetNetwork);
            return 1;
        }

        // set optimizer parameters
        optimizer->momentumCoefficient = MOMENTUM_COEFFICIENT;
        optimizer->RMSPropDecay = RMS_DECAY_RATE;

        // apply optimizer to the network
        applyOptimizer(targetNetwork, optimizer);
    } else {
        printf("Invalid input\n");
        return 1;
    }


    // set up sockets for data communication between network program and game
    printf("Initializing socket communication\n");

    char buffer[BUFFER_SIZE];
    SocketInfo *sockInfo = connectToClient();
    if (sockInfo == NULL) {
        perror("Error connecting to client");
        freeNetwork(qNetwork);
        freeNetwork(targetNetwork);
        return 1;
    }

    printf("Client connected\n");

    // send game constants to client
    // format: {width,height}
    printf("Sending game constants\n");
    int width = WIDTH;
    int height = HEIGHT;
    formatGameConstants(buffer, WIDTH, HEIGHT);
    if (send(sockInfo->clientSock, buffer, strlen(buffer), 0) < 0) {
        perror("Failed to send game constants");
        freeNetwork(qNetwork);
        freeNetwork(targetNetwork);
        disconnectFromClient(sockInfo);
        return 1;
    }
    printf("Sent game constants: %s\n", buffer);



    // train the network by playing the game
    printf("Training network...\n\n");


    // declare variables for the game

    // this array keeps track of the State, Action, and Reward of a certian timestep.
    ReplayBuffer *replayBuffer = initializeBuffer(REPLAY_BUFFER_SIZE, Q_NETWORK_INPUTS);

    // structs to store the current state and next state
    GameState state = {0};
    GameState nextState = {0};

    // flag to signal the end of the game/episode
    // 0: game is still going, 1: the game has ended
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
            // set next game state from the game socket (ensuring the data is new first)
            do {
                // recieve game state
                int bytes = recv(sockInfo->clientSock, buffer, sizeof(buffer) - 1, 0);
                if (bytes < 0) {
                    perror("Failed to recieve game state");
                } else if (bytes > 0) {
                    buffer[bytes] = '\0';
                }

                // parse recieved game state
                // the format is as follows: {networkPaddleY,opponentPaddleY,networkPoints,opponentPoints,ballX,ballY,ballVelocityX,ballVelocityY}
                sscanf(buffer, "{%f,%f,%d,%d,%f,%f,%f,%f}", &nextState.netPaddleY, &nextState.oppPaddleY, &nextState.networkPoints, &nextState.opponentPoints, &nextState.ballX, &nextState.ballY, &nextState.ballVelX, &nextState.ballVelY);

                // normalize the state between -1 and 1 using game constants like width and height.
                nextState.netPaddleY = normalizeFloat(nextState.netPaddleY, 0.0f, (float)height);
                nextState.oppPaddleY = normalizeFloat(nextState.oppPaddleY, 0.0f, (float)height);
                nextState.ballX = normalizeFloat(nextState.ballX, 0.0f, (float)width);
                nextState.ballY = normalizeFloat(nextState.ballY, 0.0f, (float)height);
                nextState.ballVelX = normalizeFloat(nextState.ballVelX, -5.0f, 5.0f);
                nextState.ballVelY = normalizeFloat(nextState.ballVelY, -5.0f, 5.0f);

                //usleep(1000); // wait for 1ms to let the cpu breathe
            } while (nextState.netPaddleY == state.netPaddleY && nextState.oppPaddleY == state.oppPaddleY && nextState.networkPoints == state.networkPoints && nextState.opponentPoints == state.opponentPoints && nextState.ballX == state.ballX && nextState.ballY == state.ballY && nextState.ballVelX == state.ballVelX && nextState.ballVelY == state.ballVelY);

            if (moveCount == 0) printf("Game #%d started\n", episode + 1);

            // calculate the reward for state based on the results of the network's action (nextState)
            if (moveCount > 0) {
                calculateReward(&state, &nextState, moveCount);
            }

            // translate nextState and state to arrays that can be parsed by the neural network
            float stateArr[Q_NETWORK_INPUTS];
            float nextStateArr[Q_NETWORK_INPUTS];
            translateState(stateArr, &state);
            translateState(nextStateArr, &nextState);

            // feed the next game state through the q network (take action for next state)
            feedForward(qNetwork, nextStateArr);


            // generate random number for greedy epsilon
            // if the random number is less than the greedy epsilon, then treat it as a random move
            // otherwise, treat it as the network's decision
            if ((float)rand() / (float)RAND_MAX <= greedyEpsilon) {
                float randNum = (float)rand() / (float)RAND_MAX;
                if (randNum <= 0.3f) {
                    nextState.move = 0.0f;
                } else if (randNum >= 0.6f) {
                    nextState.move = 1.0f;
                } else {
                    nextState.move = 0.5f;
                }
            } else {
                // get the actor network's decision and store it in the decisions array
                // the selected move is the highest value output (argmax of the output layer)
                // index 0 is up, index 1 is down, index 2 is stay still
                int maxIndex = floatArgmax(qNetwork->layers[qNetwork->layerCount - 1].values, Q_NETWORK_OUTPUTS);
                if (maxIndex == 0) {
                    nextState.move = 1.0f;
                } else if (maxIndex == 1) {
                    nextState.move = 0.0f;
                } else {
                    nextState.move = 0.5f;
                }
            }

            // append state to the replay buffer
            if (moveCount > 0) {
                addTransition(replayBuffer, stateArr, state.move, state.reward, endGame);
            }

            // set endGame flag to 1 if a point has been scored only if points are greater than 0
            // or if reached MAX_MOVES
            // specifically here so it doesnt interfere with the addTransition above (this is the endGame flag for nextState)
            if (nextState.networkPoints != state.networkPoints || nextState.opponentPoints != state.opponentPoints || moveCount == MAX_MOVES-1) {
                endGame = 1;
            }

            // send move to python program with the format {move}
            // format: {1.000000} for up, {0.000000} for down, {0.500000} for stay still
            formatMove(buffer, nextState.move);
            if (send(sockInfo->clientSock, buffer, strlen(buffer), 0) < 0) {
                perror("Failed to send move");
            }


            if (endGame) {
                // add next state to the replay buffer
                addTransition(replayBuffer, nextStateArr, nextState.move, nextState.reward, endGame); // basically no reward here lol

                // set previous variables
                state = nextState;
                nextState.move = 0;
                nextState.reward = 0;

                // update count vars
                if (stepsSinceBeginning >= START_TRAINING) {
                    stepCountSinceQUpdate++;
                    stepCountSinceTargetUpdate++;
                }

                episodeMoveCount++;
                stepsSinceBeginning++;
                endGame = 0;    // reset the endGame flag
                break;          // end game
            }

            // shift states over
            state = nextState;

            // clear current variables
            nextState.move = 0;
            nextState.reward = 0;

            // update count vars
            if (stepsSinceBeginning >= START_TRAINING) {
                stepCountSinceQUpdate++;
                stepCountSinceTargetUpdate++;
            }

            episodeMoveCount++;
            stepsSinceBeginning++;


            // update Q network every Q_UPDATE_FREQUENCY steps (only when the replay buffer has at least two transitions in it and once training begins)
            if (stepCountSinceQUpdate >= Q_UPDATE_FREQUENCY && replayBuffer->size > 1 && stepsSinceBeginning >= START_TRAINING) {
                float targetQValue = 0.0f;                      // the target q value for the state
                int actionTakenCurrentState = 0;                // the index of the action taken in the current state
                float backPropagatedValues[Q_NETWORK_OUTPUTS];  // the backpropagated values for the output layer

                // process the batches (calculate gradients over the whole batch)
                for (int i = 0; i < MINI_BATCH_SIZE; i++) {
                    // get random index for batch
                    int randBufferIndex = rand() % (replayBuffer->size-1);

                    // calculate the best action for the next state, according to the online network
                    feedForward(qNetwork, replayBuffer->buffer[randBufferIndex+1].state);
                    int bestNextAction = floatArgmax(qNetwork->layers[qNetwork->layerCount - 1].values, Q_NETWORK_OUTPUTS);

                    // calculate the best action for the next state, according to the target network
                    feedForward(targetNetwork, replayBuffer->buffer[randBufferIndex+1].state);

                    // set index of the action taken in the current state in the replay buffer for use later
                    if (replayBuffer->buffer[randBufferIndex].action == 1.0f) {
                        actionTakenCurrentState = 0;
                    } else if (replayBuffer->buffer[randBufferIndex].action == 0.0f) {
                        actionTakenCurrentState = 1;
                    } else if (replayBuffer->buffer[randBufferIndex].action == 0.5f) {
                        actionTakenCurrentState = 2;
                    } else {
                        printf("Unexpected move value\n");
                        continue;
                    }

                    // reinforce online network actions that produce the best possible reward in the next state, predicted by the target network
                    if (replayBuffer->buffer[randBufferIndex+1].done == 1) {
                        // special case if terminal state
                        targetQValue = replayBuffer->buffer[randBufferIndex+1].reward;
                    } else {
                        // the target network evaluates the online network's choice of move for the next state and the reward is applied
                        // target_value = rₜ + γQ_target(sₜ₊₁, argmax(Q_online(sₜ₊₁, aₜ)))
                        targetQValue = replayBuffer->buffer[randBufferIndex+1].reward + DISCOUNT_FACTOR * targetNetwork->layers[targetNetwork->layerCount - 1].values[bestNextAction];
                    }

                    // setup backpropagated value array to be used in backpropagation
                    feedForward(qNetwork, replayBuffer->buffer[randBufferIndex].state);
                    for (int i = 0; i < Q_NETWORK_OUTPUTS; i++) {
                        if (i == actionTakenCurrentState){
                            backPropagatedValues[i] = targetQValue;
                        } else {
                            backPropagatedValues[i] = qNetwork->layers[qNetwork->layerCount - 1].values[i];
                        }
                    }

                    // update online network
                    computeGradients(qNetwork, backPropagatedValues);
                }

                // average the accumulated batch gradients, update parameters, & zero the gradients to prepare for next batch
                averageAccumulatedGradients(qNetwork, MINI_BATCH_SIZE);
                SGDUpdate(qNetwork);
                zeroGradients(qNetwork);

                stepCountSinceQUpdate = 0;
            }

            // update the target network every TARGET_UPDATE_FREQUENCY episodes (only once training begins)
            if (stepCountSinceTargetUpdate >= TARGET_UPDATE_FREQUENCY && stepsSinceBeginning >= START_TRAINING) {
                // copy the q network to the target network
                copyNetwork(targetNetwork, qNetwork);
                stepCountSinceTargetUpdate = 0;
            }


            // update the greedy epsilon value only if training has started
            if (greedyEpsilon > GREEDY_EPSILON_END && stepsSinceBeginning >= START_TRAINING) {
                greedyEpsilon -= (GREEDY_EPSILON_START - GREEDY_EPSILON_END) / GREEDY_EPSILON_DECAY_STEPS;
            } else if (greedyEpsilon < GREEDY_EPSILON_END && stepsSinceBeginning >= START_TRAINING) {
                greedyEpsilon = GREEDY_EPSILON_END; // clamping in case we go below GREEDY_EPSILON_END
            }
        }

        printf("Game #%d completed\n", episode + 1);

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
        printf("Greedy Epsilon: %f\n", greedyEpsilon);
        printf("Timesteps taken for episode: %d\n\n", episodeMoveCount);

        // write discounted reward to file for graphing
        if (episode % 10 == 0) { 
            fprintf(discountedRewardFile, "%d, %f\n", episode, totalDiscountedReward);
        }

        // reset the move count for the current episode
        episodeMoveCount = 0;
    }

    // finished training
    printf("Finished training\n");
    // play cool sound :D
    system("afplay /System/Library/Sounds/Blow.aiff ");


    // ask user if they would like to save the network
    char save = '\0';
    while ((save = getchar()) != '\n'); // consume the newline character

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

            // free networks and game states
            freeNetwork(qNetwork);
            freeNetwork(targetNetwork);
            freeBuffer(replayBuffer);

            disconnectFromClient(sockInfo);
            
            return 0;
        } else if (save == 'n') {
            printf("Network not saved\n");
            
            // free networks and game states
            freeNetwork(qNetwork);
            freeNetwork(targetNetwork);
            freeBuffer(replayBuffer);

            disconnectFromClient(sockInfo);

            return 0;
        } else {
            printf("Invalid input\n");
        }
    } while (save != 'y' && save != 'n');


    // run the network against the opponent
    playGame:

    sockInfo = connectToClient();
    if (sockInfo == NULL) {
        perror("Error connecting to client");
        freeNetwork(qNetwork);
        freeNetwork(targetNetwork);
        return -1;
    }

    // send game constants to client
    // format: {width,height}
    printf("Sending game constants\n");
    width = WIDTH;
    height = HEIGHT;
    formatGameConstants(buffer, WIDTH, HEIGHT);
    if (send(sockInfo->clientSock, buffer, strlen(buffer), 0) < 0) {
        perror("Failed to send game constants");
        freeNetwork(qNetwork);
        freeNetwork(targetNetwork);
        disconnectFromClient(sockInfo);
        return 1;
    }
    printf("Sent game constants: %s\n", buffer);

    // clear relevant game variables
    printf("Clearing game variables\n");
    state = (GameState){0};
    nextState = (GameState){0};

    printf("Running network against opponent (please start the python game in another window (python3 game.py))\n");

    // run game loop indefinitely
    while (1) {
        // set next game state from state files (ensuring the data is new first)
        do {
            // recieve game state
            int bytes = recv(sockInfo->clientSock, buffer, sizeof(buffer) - 1, 0);
            if (bytes < 0) {
                perror("Failed to recieve game state");
            } else if (bytes > 0) {
                buffer[bytes] = '\0';
            }

            // parse recieved game state
            // the format is as follows: {networkPaddleY,opponentPaddleY,networkPoints,opponentPoints,ballX,ballY,ballVelocityX,ballVelocityY}
            sscanf(buffer, "{%f,%f,%d,%d,%f,%f,%f,%f}", &nextState.netPaddleY, &nextState.oppPaddleY, &nextState.networkPoints, &nextState.opponentPoints, &nextState.ballX, &nextState.ballY, &nextState.ballVelX, &nextState.ballVelY);

            // normalize the state between -1 and 1 using game constants like width and height, or for ball velocity, -5 and 5.
            nextState.netPaddleY = normalizeFloat(nextState.netPaddleY, 0.0f, (float)height);   // network paddle y
            nextState.oppPaddleY = normalizeFloat(nextState.oppPaddleY, 0.0f, (float)height);   // opponent paddle y
            nextState.ballX = normalizeFloat(nextState.ballX, 0.0f, (float)width);              // ball x
            nextState.ballY = normalizeFloat(nextState.ballY, 0.0f, (float)height);             // ball y
            nextState.ballVelX = normalizeFloat(nextState.ballVelX, -5.0f, 5.0f);               // ball velocity x
            nextState.ballVelY = normalizeFloat(nextState.ballVelY, -5.0f, 5.0f);               // ball velocity y
        } while (nextState.netPaddleY == state.netPaddleY && nextState.oppPaddleY == state.oppPaddleY && nextState.networkPoints == state.networkPoints && nextState.opponentPoints == state.opponentPoints && nextState.ballX == state.ballX && nextState.ballY == state.ballY && nextState.ballVelX == state.ballVelX && nextState.ballVelY == state.ballVelY);

        // translate nextState and state to arrays that can be parsed by the neural network
        float stateArr[Q_NETWORK_INPUTS];
        float nextStateArr[Q_NETWORK_INPUTS];
        translateState(stateArr, &state);
        translateState(nextStateArr, &nextState);

        // feed the input through the network
        feedForward(qNetwork, nextStateArr);

        // get the online network's decision and store it in the decisions array
        // the selected move is the highest value output (argmax of the output layer)
        // index 0 is up, index 1 is down, index 2 is stay still
        int maxIndex = floatArgmax(qNetwork->layers[qNetwork->layerCount - 1].values, Q_NETWORK_OUTPUTS);
        if (maxIndex == 0) {
            nextState.move = 1.0f;
        } else if (maxIndex == 1) {
            nextState.move = 0.0f;
        } else {
            nextState.move = 0.5f;
        }

        // send move to python program with the format {move}
        // format: {1.000000} for up, {0.000000} for down, {0.500000} for stay still
        formatMove(buffer, nextState.move);
        if (send(sockInfo->clientSock, buffer, strlen(buffer), 0) < 0) {
            perror("Failed to send move");
        }

        // set previous variables
        state = nextState;

        // clear current variables
        nextState.move = 0;
    }

    // free networks and game states
    freeNetwork(qNetwork);
    freeNetwork(targetNetwork);
    freeBuffer(replayBuffer);

    // close the connection to the client
    disconnectFromClient(sockInfo);

    return 0;
}

// clear && clear && gcc main.c -L/usr/local/lib -lneuralNetworkLib -I/usr/local/include -framework Accelerate -o main && ./main

// or for debugging:
// clear && clear && gcc main.c -L/usr/local/lib -lneuralNetworkLib -I/usr/local/include -g -framework Accelerate -o main && lldb ./main