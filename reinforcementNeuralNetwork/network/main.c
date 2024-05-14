#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#include "neuralNetworkLib/neuralNetwork.h"

// By: Rowan Rothe

// declare the q network constants (same for both q network and target network)
#define Q_NETWORK_INPUTS 6 // network paddle y position, the current opponent paddle y position, the ball x and y position, and the ball x and y velocity. this comes to 6 inputs
#define Q_NETWORK_HIDDEN_LAYERS 1 // x hidden layers for this network
#define Q_NETWORK_HIDDEN_NODES (int[]){64, 64} // x hidden layers with x nodes in each layer
#define Q_NETWORK_OUTPUTS 2 // up & down (index 0 is up, index 1 is down)
#define Q_NETWORK_LEARNING_RATE 0.0003f // learning rate for gradient descent

#define Q_NETWORK_HIDDEN_ACTIVATION 1 // 0 for sigmoid, 1 for relu, 2 for tanh, 3 for linear
#define Q_NETWORK_OUTPUT_ACTIVATION 3 // 0 for sigmoid, 1 for relu, 2 for tanh, 3 for linear, 4 for softmax

#define TARGET_UPDATE_FREQUENCY 10 // the frequency (in episodes) at which the target network is updated with the q network

// this discount factor is the amount we will decrease every timestep's reward by. its a value between 0.9 and 0.99 and is applied to every value more and more as time goes on
// for example, if the rewards of 4 timesteps are 20, 3, -10, and 5, we apply the discount factor like this: the first reward is multiplied by the discount factor once (20 * discountFactor), then the second reward is multiplied by the discount factor twice (3 * discountFactor * discountFactor), then the third reward is multiplied by the discount factor three times (-10 * discountFactor * discountFactor * discountFactor), etc...
#define DISCOUNT_FACTOR 0.99f

// the greedy epsilon value is the amount of randomness the network will have in its actions
// the network will have a 1 - greedyEpsilon chance of choosing the action with the highest value and a greedyEpsilon chance of choosing a random action
// this value will start at GREEDY_EPSILON_START and decay to GREEDY_EPSILON_END by GREEDY_EPSILON_DECAY every episode
#define GREEDY_EPSILON_START 1.0f
#define GREEDY_EPSILON_END 0.01f
#define GREEDY_EPSILON_DECAY 0.01f

#define EPISODES 1000 // the number of batches to be trained on

#define MAX_MOVES 10000 // the maximum number of moves the actor network can make before the game ends

// game constants
#define WIDTH 800
#define HEIGHT 600

// implement dropout or add noise maybe. reward the network if it chooses to move towards the ball. for example, if the ball is up and the paddle is below the ball, reward the network for choosing up.

// start with randomly choosing states, actions, rewards, and next states and train it on that
// the target value is the reward + gamma * the estimated value of the action chosen in a certain state by the online network
// 

int main() {
    // declare game variables
    FILE *moveFile; // file format: {moveValue}
    FILE *gameStateDataFile; // file format: {networkPaddleY,opponentPaddleY,networkPoints,opponentPoints,ballX,ballY,ballVelocityX,ballVelocityY}
    FILE *gameConstantsFile; // file format: {width,height}
    
    // declare network variables for the q network and target network
    Network *qNetwork;
    Network *targetNetwork;


    // seed random number generator
    printf("Seeding random number generator\n");
    srand(time(NULL));


    // clear move.txt, gameStateData.txt, and gameConstants.txt files
    printf("Clearing move.txt, gameStateData.txt, and gameConstants.txt files\n");
    moveFile = fopen("../data/move.txt", "w");
    if (moveFile == NULL) {
        perror("Error opening file");
        return 1;
    }
    fclose(moveFile);

    gameStateDataFile = fopen("../data/gameStateData.txt", "w");
    if (gameStateDataFile == NULL) {
        perror("Error opening file");
        return 1;
    }
    fclose(gameStateDataFile);

    gameConstantsFile = fopen("../data/gameConstants.txt", "w");
    if (gameConstantsFile == NULL) {
        perror("Error opening file");
        return 1;
    }
    fclose(gameConstantsFile);


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

        // consume the newline character
        while ((load = getchar()) != '\n');

        printf("Would you like to continue training the network (otherwise the program will begin playing the game)? (y/n): ");
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
    } else if (load != 'n') {
        printf("Invalid input\n");
        return 1;
    }


    // open the gameConstants.txt file
    // this file will be used to tell the python game the constants for the game
    // the format is as follows: {width,height}
    printf("Opening gameConstants.txt file\n");
    gameConstantsFile = fopen("../data/gameConstants.txt", "w");
    if (gameConstantsFile == NULL) {
        perror("Error opening file");
        return 1;
    }

    // write the game constants to the file
    int width = WIDTH;
    int height = HEIGHT;
    fprintf(gameConstantsFile, "{%d,%d}\n", width, height);
    fclose(gameConstantsFile);

    // open the gameStateData.txt file
    // this file tells the neural network the network paddle y position, the opponent paddle y position, the amount of points the network/player has, the amount of points the opponent has, the ball x and y position, and the ball x and y velocity
    // the format is as follows: {networkPaddleY,opponentPaddleY,networkPoints,opponentPoints,ballX,ballY,ballVelocityX,ballVelocityY}
    printf("Opening gameStateData.txt file\n");
    gameStateDataFile = fopen("../data/gameStateData.txt", "r");
    if (gameStateDataFile == NULL) {
        perror("Error opening file");
        return 1;
    }


    // create Q and target networks
    if (load == 'n') {
        printf("Creating Q network\n");
        qNetwork = createNetwork(Q_NETWORK_INPUTS, Q_NETWORK_HIDDEN_LAYERS, Q_NETWORK_HIDDEN_NODES, Q_NETWORK_OUTPUTS);
        if (qNetwork == NULL) {
            perror("Error creating Q network");
            return 1;
        }

        printf("Creating target network\n");
        targetNetwork = createNetwork(Q_NETWORK_INPUTS, Q_NETWORK_HIDDEN_LAYERS, Q_NETWORK_HIDDEN_NODES, Q_NETWORK_OUTPUTS);
        if (targetNetwork == NULL) {
            perror("Error creating target network");
            return 1;
        }
    }

    // train the network by playing the game
    printf("Training network... (please start the python game in another window (python3 game.py))\n\n");


    // declare variables for the game

    // this is an array to keep track of the moves the network makes per episode
    // the amount of moves is determined by the MAX_MOVES constant. if the game goes on for too long, the network will be penalized and the game will end
    // 1 is up, 0 is down
    float *decisions = malloc(MAX_MOVES * sizeof(float));
    if (decisions == NULL) {
        perror("Error allocating memory");
        return 1;
    }

    // this is an array used to keep track of the current game state which aligns with the decisions array
    // we will exclude the points as they are not needed for the network to make a decision
    // it will give the network context for the decisions it made so it knows what to improve on (done during backpropagation)
    float **gameStates = malloc(MAX_MOVES * sizeof(float *));
    if (gameStates == NULL) {
        perror("Error allocating memory");
        return 1;
    }
    for (int i = 0; i < MAX_MOVES; i++) {
        gameStates[i] = malloc(Q_NETWORK_INPUTS * sizeof(float));
        if (gameStates[i] == NULL) {
            perror("Error allocating memory");
            return 1;
        }
    }

    // this array is used to store the reward or punishment for each move the network makes at each timestep
    float *rewards = malloc(MAX_MOVES * sizeof(float));
    if (rewards == NULL) {
        perror("Error allocating memory");
        return 1;
    }

    // these variables are used to store the network paddle y position, the opponent paddle y position, the amount of points the network has, the amount of points the opponent has, the ball x and y position, and the ball x and y velocity
    int networkPaddleY = 0;
    int opponentPaddleY = 0;
    int networkPoints = 0;
    int opponentPoints = 0;
    int ballX = 0;
    int ballY = 0;
    int ballVelocityX = 0;
    int ballVelocityY = 0;

    // these variables are used to store the previous network paddle y position, the previous opponent paddle y position, the previous amount of points the network has, the previous amount of points the opponent has, the previous ball x and y position, and the previous ball x and y velocity
    int prevNetworkPaddleY = 0;
    int prevOpponentPaddleY = 0;
    int prevNetworkPoints = 0;
    int prevOpponentPoints = 0;
    int prevBallX = 0;
    int prevBallY = 0;
    int prevBallVelocityX = 0;
    int prevBallVelocityY = 0;

    // this variable is to represent the move the network has made
    // it will be written to the move.txt file like {move}
    // initialize to either 0 or 1 randomly
    float move = (float)rand() / RAND_MAX < 0.5f ? 0.0f : 1.0f;

    // this variable is used to keep track of how many moves the network has made since the last point was scored (beginning of the episode/epoch)
    int episodeMoveCount = 0;

    // this variable is used to keep track of the greedy epsilon value
    float greedyEpsilon = GREEDY_EPSILON_START;

    // this variable is used to signal the end of the game
    // 0 means the game is still going, 1 means the game has ended
    int endGame = 0;

    // for each episode, the network will play the game until a point is scored
    // after a point is scored/an episode has ended, the network will be rewarded or punished based on the moves it made
    for (int episode = 0; episode < EPISODES; episode++) {
        // repeat until a point is scored or the network has made the maximum amount of moves
        for (int moveCount = 0; moveCount < MAX_MOVES; moveCount++) {
            // read the current game state and make sure the data is new
            do {
                fseek(gameStateDataFile, 0, SEEK_SET);
                fscanf(gameStateDataFile, "{%d,%d,%d,%d,%d,%d,%d,%d}\n", &networkPaddleY, &opponentPaddleY, &networkPoints, &opponentPoints, &ballX, &ballY, &ballVelocityX, &ballVelocityY);
            } while (networkPaddleY == prevNetworkPaddleY && opponentPaddleY == prevOpponentPaddleY && networkPoints == prevNetworkPoints && opponentPoints == prevOpponentPoints && ballX == prevBallX && ballY == prevBallY && ballVelocityX == prevBallVelocityX && ballVelocityY == prevBallVelocityY);

            if (moveCount == 0) printf("Game #%d started\n", episode + 1);

            // set the gameStates array to the current game state (exclude the points as they are not needed for the network to make a decision)
            gameStates[episodeMoveCount][0] = networkPaddleY;
            gameStates[episodeMoveCount][1] = opponentPaddleY;
            gameStates[episodeMoveCount][2] = ballX;
            gameStates[episodeMoveCount][3] = ballY;
            gameStates[episodeMoveCount][4] = ballVelocityX;
            gameStates[episodeMoveCount][5] = ballVelocityY;

            // normalize the input between 0 and 1 using game constants like width and height.
            // in the case of ball velocity, it stays between -5 and 5, so normalize it between -1 and 1
            /*gameStates[episodeMoveCount][0] = (float)gameStates[episodeMoveCount][0] / (float)height;
            gameStates[episodeMoveCount][1] = (float)gameStates[episodeMoveCount][1] / (float)height;
            gameStates[episodeMoveCount][2] = (float)gameStates[episodeMoveCount][2] / (float)width;
            gameStates[episodeMoveCount][3] = (float)gameStates[episodeMoveCount][3] / (float)height;
            gameStates[episodeMoveCount][4] = (float)gameStates[episodeMoveCount][4] / 5.0f;
            gameStates[episodeMoveCount][5] = (float)gameStates[episodeMoveCount][5] / 5.0f;*/


            // break if a point has been scored
            // also make sure the points are both greater than 0
            if (networkPoints > 0 || opponentPoints > 0) {
                if (networkPoints != prevNetworkPoints || opponentPoints != prevOpponentPoints) {
                    // update the move count
                    episodeMoveCount++;

                    // end the game
                    endGame = 1;
                }
            }

            // set the reward here to the last move because we are trying to see what the reward was for the move made in the last timestep and we cant do that until we get an updated game state from the environment
            if (moveCount > 0) {
                // reward the network for hitting the ball back
                // punish it if the opponent hits the ball back
                if (ballVelocityX < prevBallVelocityX && ballVelocityX < 0 && ballX > width / 2) {
                    rewards[episodeMoveCount-1] += 1.0f;
                } else if (ballVelocityX > prevBallVelocityX && ballVelocityX > 0 && ballX < width / 2) {
                    rewards[episodeMoveCount-1] += -1.0f;
                }

                // reward the network if the network has scored a point
                // punish it if the opponent has scored a point
                if (networkPoints > 0 || opponentPoints > 0) {
                    if (networkPoints != prevNetworkPoints || opponentPoints != prevOpponentPoints) {
                        if (networkPoints > prevNetworkPoints) {
                            rewards[episodeMoveCount-1] += 10.0f;
                        } else if (opponentPoints > prevOpponentPoints) {
                            rewards[episodeMoveCount-1] += -10.0f;
                        }
                    }
                }

                // punish the network for trying to move outside of the game bounds
                if ((networkPaddleY == 0 && decisions[episodeMoveCount-1] == 1.0f) || (networkPaddleY == height - 100 && decisions[episodeMoveCount-1] == 0.0f)) {
                    rewards[episodeMoveCount-1] += -0.1f;
                }

                // reward the network for moving in the direction of the ball
                // punish otherwise
                /*if (gameStates[episodeMoveCount-1][3] - gameStates[episodeMoveCount-1][0] > 0 && decisions[episodeMoveCount-1] == 0.0f) {
                    rewards[episodeMoveCount-1] += 2.0f;
                } else if (gameStates[episodeMoveCount-1][3] - gameStates[episodeMoveCount-1][0] < 0 && decisions[episodeMoveCount-1] == 0.0f) {
                    rewards[episodeMoveCount-1] += 2.0f;
                } else {
                    rewards[episodeMoveCount-1] += -5.0f;
                }*/

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


            // check if the game has ended
            if (endGame) {
                // end the game
                endGame = 0;
                break;
            }
            

            // feed the current game state through the q network
            feedForward(qNetwork, gameStates[episodeMoveCount], Q_NETWORK_HIDDEN_ACTIVATION, Q_NETWORK_OUTPUT_ACTIVATION);


            // generate random number for greedy epsilon
            // if the random number is less than the greedy epsilon, then treat it as a random move
            // otherwise, treat it as the network's decision
            if ((float)rand() / RAND_MAX < greedyEpsilon) {
                move = ((float)rand() / RAND_MAX < 0.5f) ? 0.0f : 1.0f;
            } else {
                // get the actor network's decision and store it in the decisions array
                // the selected move is the highest value output (argmax of the output layer)
                // index 0 is up, index 1 is down
                if (qNetwork->layers[qNetwork->layerCount - 1].values[0] > qNetwork->layers[qNetwork->layerCount - 1].values[1]) {
                    move = 1.0f;
                } else {
                    move = 0.0f;
                }
            }

            // store the decision in the decisions array
            decisions[episodeMoveCount] = move;
            //printf("decision: %f\n", move);


            // write the move to the move.txt file with the format {move}
            // open the move.txt file
            // this file will be used to tell the python script what move the network would like to make
            // the format is as follows: {1} for up, {0} for down
            moveFile = fopen("../data/move.txt", "w");
            if (moveFile == NULL) {
                perror("Error opening file");
                return 1;
            }
            fseek(moveFile, 0, SEEK_SET);
            fprintf(moveFile, "{%f}", move);
            fclose(moveFile);


            // set the previous data to the current data
            prevNetworkPaddleY = networkPaddleY;
            prevOpponentPaddleY = opponentPaddleY;
            prevNetworkPoints = networkPoints;
            prevOpponentPoints = opponentPoints;
            prevBallX = ballX;
            prevBallY = ballY;
            prevBallVelocityX = ballVelocityX;
            prevBallVelocityY = ballVelocityY;

            // update the move count
            episodeMoveCount++;
        }

        printf("Game #%d completed\n", episode + 1);

        // variables to train the q network and target network
        //float qValue = 0.0f;                          // the q value/selected action for the current state
        //int qValueIndex = 0;                          // the index of the qValue in terms of output layer indexes
        float targetQValue = 0.0f;                      // the target q value for the current state
        int targetQValueIndex = 0;                      // the index of the target q value
        float backPropagatedValues[Q_NETWORK_OUTPUTS];  // the backpropagated values for the output layer
        float totalDiscountedReward = 0.0f;             // the total discounted reward for the episode
        int randomCurrentTimestep = 0;                  // the random current timestep to train the network on
        int randomNextTimestep = 0;                     // the random next timestep to train the network on

        // calculate the total discounted reward for the episode
        for (int i = 0; i < episodeMoveCount; i++) {
            totalDiscountedReward = rewards[i] + DISCOUNT_FACTOR * totalDiscountedReward;
        }

        /*// train the q network every episode but only update the target network every TARGET_UPDATE_FREQUENCY episodes
        // randomly sample from the saved game states and rewards and train the network on them
        for (int currentTimestep = 0; currentTimestep < episodeMoveCount; currentTimestep++) {
            if (currentTimestep < episodeMoveCount - 1) {
                // calculate the q value for the current state // feed forward the q network to update the network for later backpropagation
                feedForward(qNetwork, gameStates[currentTimestep], Q_NETWORK_HIDDEN_ACTIVATION, Q_NETWORK_OUTPUT_ACTIVATION);
                /*if (decisions[currentTimestep] == 1.0f) {
                    qValue = qNetwork->layers[qNetwork->layerCount - 1].values[0];
                    qValueIndex = 0;
                } else {
                    qValue = qNetwork->layers[qNetwork->layerCount - 1].values[1];
                    qValueIndex = 1;
                }fix me here lol. its supposed to be star forward slash

                // calculate the target q value for the current state
                feedForward(targetNetwork, gameStates[currentTimestep], Q_NETWORK_HIDDEN_ACTIVATION, Q_NETWORK_OUTPUT_ACTIVATION);
                if (decisions[currentTimestep] == 1.0f) {
                    targetQValue = targetNetwork->layers[targetNetwork->layerCount - 1].values[0];
                    targetQValueIndex = 0;
                } else {
                    targetQValue = targetNetwork->layers[targetNetwork->layerCount - 1].values[1];
                    targetQValueIndex = 1;
                }

                // apply reward/punishment to the target q value to get the ideal q value
                targetQValue = rewards[currentTimestep] + DISCOUNT_FACTOR * targetQValue;
            } else {
                feedForward(qNetwork, gameStates[currentTimestep], Q_NETWORK_HIDDEN_ACTIVATION, Q_NETWORK_OUTPUT_ACTIVATION);
                targetQValue = rewards[currentTimestep];
            }

            // setup the backpropagated values array to be used in backpropagation
            for (int i = 0; i < Q_NETWORK_OUTPUTS; i++) {
                if (i == targetQValueIndex) {
                    backPropagatedValues[i] = targetQValue;
                } else {
                    backPropagatedValues[i] = qNetwork->layers[qNetwork->layerCount - 1].values[i];
                }
            }

            // feed in the ideal q value to the q network and calculate the loss
            backPropagate(qNetwork, backPropagatedValues, Q_NETWORK_LEARNING_RATE, Q_NETWORK_HIDDEN_ACTIVATION, Q_NETWORK_OUTPUT_ACTIVATION);
        }*/

        // train the q network every episode but only update the target network every TARGET_UPDATE_FREQUENCY episodes
        // randomly sample from the saved game states and rewards and train the network on them
        for (int currentTimestep = 0; currentTimestep < episodeMoveCount; currentTimestep++) {
            // get random timestep from replay buffer and the proceeding timestep
            randomCurrentTimestep = currentTimestep;//(float)rand() / RAND_MAX * episodeMoveCount;
            randomNextTimestep = currentTimestep+1;//randomCurrentTimestep + 1;

            if (randomCurrentTimestep < episodeMoveCount - 1) {
                // feed forward the q network to update the network for later backpropagation // calculate the q value for the current state
                feedForward(qNetwork, gameStates[randomCurrentTimestep], Q_NETWORK_HIDDEN_ACTIVATION, Q_NETWORK_OUTPUT_ACTIVATION);
                /*if (decisions[currentTimestep+1] == 1.0f) {
                    qValue = qNetwork->layers[qNetwork->layerCount - 1].values[0];
                    qValueIndex = 0;
                } else {
                    qValue = qNetwork->layers[qNetwork->layerCount - 1].values[1];
                    qValueIndex = 1;
                }*/
                /*for (int i = 0; i < Q_NETWORK_OUTPUTS; i++) {
                    if (i == 0) {
                        qVaue = qNetwork->layers[qNetwork->layerCount - 1].values[i];
                        qValueIndex = i;
                    } else if (qNetwork->layers[qNetwork->layerCount - 1].values[i] > qValue) {
                        qValue = qNetwork->layers[qNetwork->layerCount - 1].values[i];
                        qValueIndex = i;
                    }
                }*/

                // calculate the target q value for the current state
                feedForward(targetNetwork, gameStates[randomNextTimestep], Q_NETWORK_HIDDEN_ACTIVATION, Q_NETWORK_OUTPUT_ACTIVATION);
                if (decisions[randomCurrentTimestep] == 1.0f) {
                    targetQValue = targetNetwork->layers[targetNetwork->layerCount - 1].values[0];
                    targetQValueIndex = 0;
                } else {
                    targetQValue = targetNetwork->layers[targetNetwork->layerCount - 1].values[1];
                    targetQValueIndex = 1;
                }

                // apply reward/punishment to the target q value to get the ideal q value
                targetQValue = rewards[randomNextTimestep] + DISCOUNT_FACTOR * targetQValue;
            } else {
                feedForward(qNetwork, gameStates[randomCurrentTimestep], Q_NETWORK_HIDDEN_ACTIVATION, Q_NETWORK_OUTPUT_ACTIVATION);
                targetQValue = rewards[randomNextTimestep];
            }

            // setup the backpropagated values array to be used in backpropagation
            for (int i = 0; i < Q_NETWORK_OUTPUTS; i++) {
                if (i == targetQValueIndex) {
                    backPropagatedValues[i] = targetQValue;
                } else {
                    backPropagatedValues[i] = qNetwork->layers[qNetwork->layerCount - 1].values[i];
                }
            }

            // feed in the ideal q value to the q network and calculate the loss
            backPropagate(qNetwork, backPropagatedValues, Q_NETWORK_LEARNING_RATE, Q_NETWORK_HIDDEN_ACTIVATION, Q_NETWORK_OUTPUT_ACTIVATION);
        }

        // update the target network every TARGET_UPDATE_FREQUENCY episodes
        if (episode % TARGET_UPDATE_FREQUENCY == 0) {
            // copy the q network to the target network
            copyNetwork(targetNetwork, qNetwork);
        }

        // print total reward
        printf("Total discounted reward: %f\n\n", totalDiscountedReward);

        // print actor network
        //printf("Actor network:\n");
        //printNetwork(actorNetwork);


        // update prev points
        prevNetworkPoints = networkPoints;
        prevOpponentPoints = opponentPoints;

        // reset the move count for the current episode
        episodeMoveCount = 0;

        // update the greedy epsilon value
        if (greedyEpsilon > GREEDY_EPSILON_END) {
            greedyEpsilon -= GREEDY_EPSILON_DECAY;
        }

        // reset the rewards array, decisions array, and game states array
        for (int i = 0; i < MAX_MOVES; i++) {
            rewards[i] = 0.0f;
            decisions[i] = 0.0f;
            for (int j = 0; j < Q_NETWORK_INPUTS; j++) {
                gameStates[i][j] = 0.0f;
            }
        }

        // reset the total reward
        totalDiscountedReward = 0;
    }

    // finished training
    printf("Finished training\n");


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
            break;
        } else if (save == 'n') {
            printf("Network not saved\n");
            break;
        } else {
            printf("Invalid input\n");
        }
    } while (save != 'y' && save != 'n');


    // run the network against the opponent
    playGame:
    /*printf("Running network against opponent (please start the python game in another window (python3 game.py))\n");

    // open the move.txt file
    moveFile = fopen("../data/move.txt", "w");
    if (moveFile == NULL) {
        perror("Error opening file");
        return 1;
    }

    // open the gameStateData.txt file
    gameStateDataFile = fopen("../data/gameStateData.txt", "r");
    if (gameStateDataFile == NULL) {
        perror("Error opening file");
        return 1;
    }

    // run game loop indefinitely
    while (1) {
        // get the network paddle y position, the opponent paddle y position, the amount of points the network has, the amount of points the opponent has, the ball x and y position, and the ball x and y velocity
        fscanf(gameStateDataFile, "{%d,%d,%d,%d,%d,%d,%d,%d}\n", &networkPaddleY, &opponentPaddleY, &networkPoints, &opponentPoints, &ballX, &ballY, &ballVelocityX, &ballVelocityY);

        // feed the network the current game state
        input[0] = networkPaddleY;
        input[1] = opponentPaddleY;
        input[2] = ballX;
        input[3] = ballY;
        input[4] = ballVelocityX;
        input[5] = ballVelocityY;

        // normalize the input between 0 and 1 using game constants like width and height.
        // in the case of ball velocity, we will make a negative value 0 and a positive value 1 (instead of 5 and -5)
        input[0] = (float)input[0] / (float)height;
        input[1] = (float)input[1] / (float)height;
        input[2] = (float)input[2] / (float)width;
        input[3] = (float)input[3] / (float)height;
        input[4] = (input[4] < 0) ? 0 : 1;
        input[5] = (input[5] < 0) ? 0 : 1;

        // feed the input through the network
        feedForward(network, input, HIDDEN_ACTIVATION, OUTPUT_ACTIVATION);

        // get the network's decision then store it in the decisions array and output it to the move.txt file
        move = network->layers[network->layerCount - 1].values[0];

        // write the move to the move.txt file
        fprintf(moveFile, "%f\n", move);
    }*/

    // free networks, decisions, and rewards
    freeNetwork(qNetwork);
    freeNetwork(targetNetwork);
    //free(decisions);
    //free(rewards);
    //free(gameStates);

    // close the files
    fclose(moveFile);
    fclose(gameStateDataFile);
    fclose(gameConstantsFile);

    return 0;
}

// clear && clear && gcc main.c neuralNetworkLib/neuralNetwork.c -Wall -Wextra -o main && ./main
