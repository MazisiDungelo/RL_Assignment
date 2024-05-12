import numpy as np
from FourRooms import FourRooms
import random

learning_rate = 0.1  # Learning rate
discountFactor = 0.99  # Discount factor

def main():
    # Initialize Q-values for all state-action pairs
    Q_TABLE = np.zeros((13,13,4))
    # Create FourRooms Object
    fourRoomsObj = FourRooms('multi')

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
    first = 0
    second = 0
    first_package_state = (0,0)
    second_package_state = (0,0)
    third_package_state = (0,0)

    while packagesRemaining > 0:    
        for episode in range(50):
            for step, act in enumerate(actSeq):
                currentState = fourRoomsObj.getPosition()
                gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(act)
                if currentState == newPos: #hit the wall
                    reward = -1
                else:
                    reward = -0.2 

                if packagesRemaining == 2 and first == 0: # found first package
                    reward = 1
                    first_package_state = currentState
                    print("Package 1 collected")
                    first += 1
                elif packagesRemaining == 1 and second == 0: # found second package
                    reward = 1
                    second_package_state = currentState
                    print("Package 2 collected")
                    second += 1
                elif packagesRemaining == 0: # found third package
                    reward = 1
                    third_package_state = currentState
                nextMaxQ = np.max(Q_TABLE[newPos[0],newPos[1]])
                Q_TABLE[(currentState[0],currentState[1],act)] += learning_rate * (reward + (discountFactor * nextMaxQ) - Q_TABLE[(currentState[0],currentState[1],act)])

                best_Q_value_index = np.argmax(Q_TABLE[(currentState[0],currentState[1])])
                actSeq[step] = best_Q_value_index
                print("Agent took {0} action and moved to {1} of type {2}".format (aTypes[act], newPos, gTypes[gridType]))


                if isTerminal:
                   print("Agent started @ {0}".format(start_state))
                   print("Package 1 collected @ {0}\nPackage 2 collected @ {1}\nPackage 3 collected @ {2}".format(first_package_state,
                                                                                          second_package_state,
                                                                                          third_package_state)) 
                   
                   break
            if isTerminal:
                break
        fourRoomsObj.newEpoch()
    
    # Show Path
    fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()