[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/15965062/44311825-11761100-a3f7-11e8-8412-5d14ee230bf7.png "Algorithm"
[image2]: https://user-images.githubusercontent.com/15965062/44312723-1c37a280-a405-11e8-8671-b811fc21d687.png "Plot of Rewards"

# Report - Deep RL Project: Navigation

### Implementation Details

The code for this project is ordered in 2 python files, 'dqn_agent.py' and 'model.py', and the main training code and instructions in the notebook 'Navigation.ipynb'. 

1. 'model.py': Architecture and logic for a neural network implementing the state to action values from which the maximum action value is chosen for the DQN algorithm.

2. 'dqn_agent.py': Implements the agent class, which includes the logic for the stepping, acting, learning and the buffer to hold the experience data on which to train the agent, and uses 'model.py' to generate the local and target networks.

3. 'Navigation.ipynb': Main training logic and usage instructions. Includes explainations about the environment, state and action space, goals and final results. The main training loop creates an agent and trains it using the double-DQN algorithm (details below) until satisfactory results. 

### Learning Algorithm

The agent is trained using the double-DQN algorithm.

References:
1. [DQN Paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

2. [Double-DQN Paper](https://arxiv.org/abs/1509.06461)

3. ![Algorithm][image1]


4. Short explanation (refer to the papers for further details):
    - DQN: Adding experience replay (to decorrelate the experiences) and fixed target network (using second network that is updated slower then the main neural network) to the original Q learning algorithm.

    - Double-DQN: Using a different network to choose the argmax action from above algorithm (and since we have the fixed target network, we just use that). This is used to somewhat fix the problem of overestimation of Q values in the original DQN algoritm.
    
5. Hyperparameters used:
    ```
    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network
    n_episodes = 1400       # max number of episodes for the training loop
    max_t = 1000            # max steps in every episode 
    eps_start=1.0           # starting value of epsilon
    eps_end=0.01            # final value of epsilon
    eps_decay=0.995          # decay rate for epsilon
    ```

6. Network architecture:
    - Fully connected network, with 2 hidden layers of 64 units each and Relu activation function.
    - Input and output layers sizes determined by the state and action space, respectively.
    - Training time until solving the environment (~350 episodes) takes less than 5 minutes on AWS p2 instance with GPU.
    - See 'model.py' for more details.

### Plot of results

As seen below, the agent solves the environment after ~350 episodes, and achieves best average score of above 17.

![Plot of Rewards][image2]

###  Ideas for future work

1. Learning from Pixels:
    - Learning from pixels instead of the given states will require a different network architecture (CNN based) and additional training time.
    - Aside from above, the code implemented here is almost ready for this challange, will only need to change the state space and load the AWS environment with X server (the provided env for Linux depends on X server).

2. Rainbow:
    - Several improvements to the DQN algorithm (in addition to Double-DQN) have risen over the years.
    - The [Rainbow paper](https://arxiv.org/abs/1710.02298) combines these ideas, adding some or all of them to this repo would be nice. 
