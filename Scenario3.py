import numpy as np
from FourRooms import FourRooms
import random

learning_rate = 0.5  # Learning rate
discountFactor = 0.99  # Discount factor

def main():
    # Initialize Q-values for all state-action pairs
    Q_TABLE = np.zeros((13,13,4))
    # Create FourRooms Object
    fourRoomsObj = FourRooms('rgb')

    # This will try to draw a zero
    actSeq = [FourRooms.LEFT, FourRooms.LEFT, FourRooms.LEFT,
              FourRooms.UP, FourRooms.UP, FourRooms.UP,
              FourRooms.RIGHT, FourRooms.RIGHT, FourRooms.RIGHT,
              FourRooms.DOWN, FourRooms.DOWN, FourRooms.DOWN]

    aTypes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    gTypes = ['EMPTY', 'RED', 'GREEN', 'BLUE']
    gridType = ''
    start_state = fourRoomsObj.getPosition()
    print('Agent starts at: {0}'.format(fourRoomsObj.getPosition()))
    packagesRemaining = 3
    package_order = ['RED','GREEN','BLUE']
    collection_index = 0
    while packagesRemaining > 0:    
        for episode in range(50):
            for step, act in enumerate(actSeq):
                currentState = fourRoomsObj.getPosition()
                gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(act)

                if currentState == newPos or gridType !=  package_order[collection_index]: # hit the wall
                    reward = -1
                elif gridType == package_order[collection_index]:
                    reward = 1
                    collection_index += 1
                else:
                    reward = -0.2 

                nextMaxQ = np.max(Q_TABLE[newPos[0],newPos[1]])
                Q_TABLE[(currentState[0],currentState[1],act)] += learning_rate * (reward + (discountFactor * nextMaxQ) - Q_TABLE[(currentState[0],currentState[1],act)])

                best_Q_value_index = np.argmax(Q_TABLE[(currentState[0],currentState[1])])
                actSeq[step] = best_Q_value_index
                print("Agent took {0} action and moved to {1} of type {2}".format (aTypes[act], newPos, gTypes[gridType]))


                if isTerminal:   
                   break
            if isTerminal:
                print("All 3 packages collected")
                break
        fourRoomsObj.newEpoch()
    
    # Show Path
    fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()