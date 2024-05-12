import numpy as np
from FourRooms import FourRooms
import random

learning_rate = 0.1  # Learning rate
discountFactor = 0.99  # Discount factor

def main():
    # Initialize Q-values for all state-action pairs
    Q_TABLE = np.zeros((13,13,4))
    # Create FourRooms Object
    fourRoomsObj = FourRooms('simple')

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
        for episode in range(2):
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

                best_Q_value_index = np.argmax(Q_TABLE[(currentState[0],currentState[1])])
                actSeq[step] = best_Q_value_index
                print("Agent took {0} action and moved to {1} of type {2}".format (aTypes[act], newPos, gTypes[gridType]))


                if isTerminal:
                    print(packageStage)
                    print("Found package")
                    break
        fourRoomsObj.newEpoch()
    
    # Show Path
    fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()