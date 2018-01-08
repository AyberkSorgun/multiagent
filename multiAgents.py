# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodList = newFood.asList()
        totalDist = 1
        ghostLocation = newGhostStates[0].getPosition()
        distanceToGhost = util.manhattanDistance(newPos, ghostLocation)
        ghostFactor = 1

        if distanceToGhost <= 3:
            ghostFactor = 0.01

        for foods in foodList:
            totalDist += util.manhattanDistance(newPos, foods)

        return (1.0 / totalDist) + scoreEvaluationFunction(successorGameState) - (1.0 / ghostFactor)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def value(gameState, agentIndex, depth):
            if depth == 1 or len(gameState.getLegalActions((agentIndex+1) % gameState.getNumAgents()))==0:
                return self.evaluationFunction(gameState)
            if agentIndex % gameState.getNumAgents() == 0:
                return maxvalue(gameState, agentIndex % gameState.getNumAgents(),depth-1)
            else:
                return minvalue(gameState, agentIndex % gameState.getNumAgents(),depth-1)

        def maxvalue(gameState, agentIndex, depth):
            val = [-999999]
            for actions in gameState.getLegalActions(agentIndex):
                val.append(value(gameState.generateSuccessor(agentIndex, actions),agentIndex+1, depth))
            return max(val)

        def minvalue(gameState, agentIndex, depth):
            val = [999999]
            for actions in gameState.getLegalActions(agentIndex):
                val.append(value(gameState.generateSuccessor(agentIndex, actions), agentIndex+1, depth))
            return min(val)
        dictionary = {}
        for actions in gameState.getLegalActions(0):
            dictionary[value(gameState.generateSuccessor(0, actions), 1, self.depth*gameState.getNumAgents())] = actions
        return dictionary[max(dictionary.iterkeys())]


        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def value(gameState, agentIndex, depth,alpha,beta):
            if depth == 1 or len(gameState.getLegalActions((agentIndex+1) % gameState.getNumAgents()))==0:
                return self.evaluationFunction(gameState)
            if agentIndex % gameState.getNumAgents() == 0:
                return maxvals(gameState, agentIndex % gameState.getNumAgents(),depth-1, alpha, beta)
            else:
                return minvals(gameState, agentIndex % gameState.getNumAgents(),depth-1, alpha,beta)

        def maxvals(gameState,agentIndex, depth, alpha, beta):
            val = -999999
            for actions in gameState.getLegalActions(agentIndex):
                val = max(val, value(gameState.generateSuccessor(agentIndex, actions), agentIndex+1, depth,alpha,beta))
                if val > beta:
                    return val
                alpha = max(alpha, val)

            return val

        def minvals(gameState, agentIndex, depth, alpha, beta):
            val = 999999
            for actions in gameState.getLegalActions(agentIndex):
                val = min(val, value(gameState.generateSuccessor(agentIndex, actions), agentIndex + 1, depth,alpha, beta))
                if val < alpha:
                    return val
                beta = min(beta, val)
            return val

        dictionary = {}
        alpha = -999999
        beta = 999999
        for actions in gameState.getLegalActions(0):
            dictionary[value(gameState.generateSuccessor(0, actions), 1, self.depth * gameState.getNumAgents(), alpha, beta)] = actions
            alpha = max(value(gameState.generateSuccessor(0, actions), 1, self.depth * gameState.getNumAgents(), alpha, beta),alpha)
        return dictionary[max(dictionary.iterkeys())]





        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def value(gameState, agentIndex, depth):
            if depth == 1 or len(gameState.getLegalActions((agentIndex+1) % gameState.getNumAgents()))==0:
                return self.evaluationFunction(gameState)
            if agentIndex % gameState.getNumAgents() == 0:
                return maxvalue(gameState, agentIndex % gameState.getNumAgents(),depth-1)

            else:
                return expvalue(gameState, agentIndex % gameState.getNumAgents(),depth-1)

        def expvalue(gameState,agentIndex,depth):
            v = 0
            for actions in gameState.getLegalActions(agentIndex):
                p = 1.0 / (len(gameState.getLegalActions(agentIndex)))
                v += p * value(gameState.generateSuccessor(agentIndex, actions),agentIndex+1, depth)

            return v

        def maxvalue(gameState, agentIndex, depth):
            val = [-999999]
            for actions in gameState.getLegalActions(agentIndex):
                val.append(value(gameState.generateSuccessor(agentIndex, actions),agentIndex+1, depth))
            return max(val)

        def minvalue(gameState, agentIndex, depth):
            val = [999999]
            for actions in gameState.getLegalActions(agentIndex):
                val.append(value(gameState.generateSuccessor(agentIndex, actions), agentIndex+1, depth))
            return min(val)
        dictionary = {}
        for actions in gameState.getLegalActions(0):
            dictionary[value(gameState.generateSuccessor(0, actions), 1, self.depth*gameState.getNumAgents())] = actions
        return dictionary[max(dictionary.iterkeys())]

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    foodList = newFood.asList()
    totalDist = 1
    ghostLocation = newGhostStates[0].getPosition()
    distanceToGhost = util.manhattanDistance(newPos, ghostLocation)
    ghostFactor = 1
    totalDistCap = 1
    if distanceToGhost <= 3:
        ghostFactor = 0.01
    if newScaredTimes > 0:
        ghostFactor = -0.01

    for foods in foodList:
        totalDist += util.manhattanDistance(newPos, foods)
    for capsules in currentGameState.getCapsules():
        totalDistCap += util.manhattanDistance(newPos, capsules)
    return (1.0 / totalDist) + scoreEvaluationFunction(currentGameState) - (1.0 / ghostFactor) + (1.0 / totalDistCap)

# Abbreviation


better = betterEvaluationFunction

