import numpy as np
from FourRooms import FourRooms
import sys

learning_rate = 0.1  # Learning rate
discountFactor = 0.99  # Discount factor

def main(stochastic):
    # Initialize Q-values for all state-action pairs
    Q_TABLE = np.zeros((13,13,4))
    # Create FourRooms Object
    fourRoomsObj = FourRooms('simple',stochastic=stochastic)

    # This will try to draw a zero
    actSeq = [FourRooms.LEFT, FourRooms.LEFT, FourRooms.LEFT,
              FourRooms.UP, FourRooms.UP, FourRooms.UP,
              FourRooms.RIGHT, FourRooms.RIGHT, FourRooms.RIGHT,
              FourRooms.DOWN, FourRooms.DOWN, FourRooms.DOWN]

    aTypes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    gTypes = ['EMPTY', 'RED', 'GREEN', 'BLUE']

    print('Agent starts at: {0}'.format(fourRoomsObj.getPosition()))
    packageStage = (0,0)
    packagesRemaining = 1
    while packagesRemaining > 0:    
        for episode in range(10):
            total_reward = 0
            for step, act in enumerate(actSeq):
                currentState = fourRoomsObj.getPosition()
                gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(act)
                if currentState == newPos: #hit the wall
                    reward = -1
                else:
                    reward = -0.2 

                if packagesRemaining == 0: # found package
                    packageStage = currentState
                    reward = 1
                nextMaxQ = np.max(Q_TABLE[newPos[0],newPos[1]])
                Q_TABLE[(currentState[0],currentState[1],act)] += learning_rate * (reward + (discountFactor * nextMaxQ) - Q_TABLE[(currentState[0],currentState[1],act)])
                
                best_action = epsilon_greedy_action(Q_TABLE, currentState, epsilon=0.1)
                actSeq[step] =  best_action
                print("Agent took {0} action and moved to {1} of type {2}".format (aTypes[act], newPos, gTypes[gridType]))


                if isTerminal:
                    print(packageStage)
                    print("Found package")
                    break
            if isTerminal:
                break
        fourRoomsObj.newEpoch()
    # Show Path
    fourRoomsObj.showPath(-1)

def epsilon_greedy_action(Q_TABLE, currentState, epsilon):
    if np.random.rand() < epsilon:
        # Explore: Choose a random action
        action = np.random.randint(0, 4)
    else:
        # Exploit: Choose the action with the highest Q-value
        action = np.argmax(Q_TABLE[(currentState[0], currentState[1])])
    return action

if __name__ == "__main__":
    main('-stochastic' in sys.argv)