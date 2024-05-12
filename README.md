# Mazisi Dungelo
# DMGMAZ001
# CSC3022F RL Assignment

# RL_Assignment
This Assignment train RL Agent to pick packages on grid world
This is done in different Scenario1

# Scenario1.py
Class Scenario1.py first initiaise a grid world, default action sequence, action types and grid types

Q_Table is initialise to 11 x 11 and in each grid cell Agent has 4 actions to take 11 x 11 x 4

The Agent is given a random starting position and begins to start training 
Training is done in 10 episodes, after these episodes, if package i not found , grid world is reset (new starting positions and new package position)

The Agent takes actions in the action sequence in each episode 

The Agent checks if there is a change in states otherwise , this is seen hitting the wall, reward is given as negative 1
if no wall is hit but not Terminal state the reward is negative small number 

Finds next Maximum Q value
The value is use to update Q_TABLE is this formula = Q(s, a) = Q(st
, at
) + 𝜶[(rt + 𝛾maxa
Q(st+1 , a)) - Q(st
, at
)]

Using epsilon the next best action is calculated using Q values in the current state
Thos actions are added in the action sequence updated the previous

This is done until Terminal state = package found

# Scenario2.py
The Agent picks 3 packages in the almost same way as in Scenario 1
Few differences are that:

The agent takes 50 episodes
The agent trains untill all three packages are found
Reward of 1 is given for each package found 

# Scenario3.py
The agent picks 3 packages in order
This act almost the same as Scenario 2
Additions:

Negative reward of -1 is given to when hitting the wall and when gridType of package is not order of collects
Otherwise reward of 1 is given

# Scenario 4
This adds stochastic flag in each of the Scenario file
Checks if command line has argument '-stochatic' if that is true else false
If True the fourRoom stochastic is set to True

# requirement.txt
Requirements for running programs

# Invokation
All three Scenarios are invoked as :
python Scenario(number).py 
or 
python Scenario(number).py -stochastic

