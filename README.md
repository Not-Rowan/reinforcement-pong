# reinforcement-pong
This is a reinforcement learning pong agent I made using my custom neural network C library.

There are many options for running this game which I have included here.

## General Info
In each file, there are hyperparameters and game parameters. These can be edited to change the behaviour of the programs. Inside each program file, there are also terminal commands to run the respective program at the bottom

## Game:
The game can be run through Python or through C. The Python version only supports a hard-coded opponent with a single bot player. The C version supports the same as the Python version but adds the ability for dueling agents (both players are agents)
Note: For the C game, you may have to install SDL2 and initialize it

## Main Program:
There are two main programs. They are essentially the same file except one has a different port (to allow it to work with the dueling agent setting)


## Running The Whole Thing
To run the whole thing, the different options above can be used. One terminal tab or window will be running the main agent program (main.c or main2.c if dueling agents), then the other window/tab will be running the game program (either game.c or game.py).

---
Hopefully this helps and contact me if you need any help with running the program. I'd be happy to help any time :)
