from pacman import Directions
from game import Agent
from util import *
from copy import deepcopy
import numpy as np
import math
from leftyghost import Leftyghost
from greedyghost import Greedyghost
# XXX: You should complete this class for Step 2

PACMAN = 0
# Coefficients
coef_next_food = -1.5
coef_act_ghost = -1/2
coef_scared_ghost = -2
coef_num_foods = -4
coef_num_caps = -20

class Agentghost2(Agent):

    def __init__(self, index=0, time_eater=40, g_pattern=0):
        """
        Arguments:
        ----------
        - `index`: index of your agent. Leave it to 0, it has been put
                   only for game engine compliancy
        - `time_eater`: Amount of time pac man remains in `eater`
                        state when eating a big food dot
        - `g_pattern`: Ghosts' pattern in-game :
                       0 - leftyghost
                       1 - greedyghost
                       2 - randyghost
                       3 - rpickyghost
        """
        # Ghosts pattern
        self.pattern = g_pattern
        self.num_agent = 0
        if self.pattern == 0:
            self.strategy = play_lefty
        elif self.pattern == 1:
            self.strategy = play_greedy
        elif self.pattern == 2:
            self.strategy = play_semi_random
        else:
            self.strategy = play_unknown

    def getAction(self, state):
        """
        Parameters:
        -----------
        - `state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)

        Return:
        -------
        - A legal move as defined game.Directions.
        """

        self.num_agent = len(state.getGhostPositions())+1
        max_depth = self.num_agent*4
        root = PacmanNode(state, max_depth)
        self.expectiminimax(root, PACMAN)

        action = None
        max_score = -math.inf
        for child in root.children:
            if child.score > max_score:
                action = child.action
                max_score = child.score

        return action


    def expectiminimax(self, node, agent):
        """
        Performs an ExpectiMinimax step.

        :param node: Node object representing the current search node
        :param agent: number representing the agent that plays at this step
        :return: the best
        """
        if node.state.isWin() or node.state.isLose():
            node.set_score(node.state.getScore())
        elif node.depth == 0:
            node.set_score(node.evaluation_function())
        else:
            if isinstance(node, PacmanNode):
                scores = []
                legal = node.state.getLegalPacmanActions()
                for action in legal:
                    state = node.state.generateSuccessor(PACMAN, action)
                    child = GhostNode(state, node.depth-1, node, action)
                    node.children.append(child)
                for child in node.children:
                    next_agent = int((agent+1) % self.num_agent)
                    scores.append(self.expectiminimax(child, next_agent))
                node.set_score(max(scores))
            else:
                score = 0
                legal = node.state.getLegalActions(agent)
                probs = self.strategy(node.state, agent)
                for action in legal:
                    state = node.state.generateSuccessor(agent, action)
                    # Next turn is Pacman
                    if agent == self.num_agent-1:
                        child = PacmanNode(state, node.depth-1, node, action)
                    else:
                        child = GhostNode(state, node.depth-1, node, action)
                    node.children.append(child)
                for child in node.children:
                    next_agent = int((agent + 1) % self.num_agent)
                    score += probs[child.action] * \
                             self.expectiminimax(child, next_agent)
                node.set_score(score)
        return node.score


###########################
#  Expecti Minimax Nodes  #
###########################

class Node:

    def __init__(self, state, depth, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.depth = depth
        self.score = 0

    def set_score(self, score):
        self.score = score

    def evaluation_function(self):

        # Get required information
        pacman = self.state.getPacmanPosition()
        ghosts = self.state.getGhostStates()
        foods = self.state.getFood().asList()
        capsules = self.state.getCapsules()
        score = self.state.getScore()

        # Compute eval functions
        next_food = closest_food(pacman, foods)
        act_ghost = closest_ghost(pacman, ghosts)
        scared = closest_scared_ghost(pacman, ghosts)
        num_food = num_foods(foods)
        num_caps = num_capsules(capsules)

        # Return score
        return score + coef_next_food*next_food + coef_act_ghost*act_ghost + \
               coef_scared_ghost*scared + coef_num_foods*num_food + \
               coef_num_caps*num_caps

class PacmanNode(Node):

    def __init__(self, state, depth, parent=None, action=None):
        super(PacmanNode, self).__init__(state, depth, parent, action)


class GhostNode(Node):

    def __init__(self, state, depth, parent=None, action=None):
        self.probs = []
        super(GhostNode, self).__init__(state, depth, parent, action)


##############################
#  Ghosts States Generation  #
##############################

def play_lefty(state, agent):
    """
    Apply the lefty pattern for a given ghost at a given state.

    :param state: GameState object representing the current state
    :param agent: number representing the ghost playing
    :return: a Counter object linking legal actions and their probability
    """

    return Leftyghost(agent).getDistribution(state)

def play_greedy(state, agent):
    """
    Apply the greedy pattern for a given ghost at a given state.

    :param state: GameState object representing the current state
    :param agent: number representing the ghost playing
    :return: a Counter object linking legal actions and their probability
    """

    return Greedyghost(agent).getDistribution(state)

def play_random(state, agent):
    """
    Apply the random pattern for a given ghost at a given state.

    :param state: GameState object representing the current state
    :param agent: number representing the ghost playing
    :return: a Counter object linking legal actions and their probability
    """
    probs = Counter()
    legal = state.getLegalActions(agent)
    for action in legal:
        probs[action] = 1/len(legal)

    return probs

def play_semi_random(state, agent):
    """
    Apply the semi random pattern for a given ghost at a given state.

    :param state: GameState object representing the current state
    :param agent: number representing the ghost playing
    :return: a Counter object linking legal actions and their probability
    """

    probs = Counter()
    legal = state.getLegalActions(agent)

    # Compute distribution for each possible strategy
    lefty = play_lefty(state, agent)
    greedy = play_greedy(state, agent)
    randy = play_random(state, agent)

    # Apply the pattern
    for action in legal:
        probs[action] = 0.25*lefty[action] + 0.5*greedy[action] + \
                        0.25*randy[action]

    return probs

def play_unknown(state, agent):
    """
    Apply all the possible patterns for a ghost for which we don't know the
    pattern at a given state.

    :param state: GameState object representing the cugrrent state
    :param agent: number representing the ghost playing
    :return: a Counter object linking legal actions and their probability
    """

    pass

###########################
#  Evaluation Functions  #
###########################

def closest_food(pacman, food_pos):
    """
    Computes the closest distance of any food dot to Pacman.

    :param pacman: tuple representing Pacman's position
    :param food_pos: list of tuples representing positions of food dots
    :return: the closest distance to Pacman
    """
    food_distances = []
    for food in food_pos:
        food_distances.append(manhattanDistance(food, pacman))
    return min(food_distances) if len(food_distances) > 0 else 1

def closest_ghost(pacman, ghosts):
    """
    Computes the closest distance of any active ghost to Pacman.

    :param pacman: tuple representing Pacman's position
    :param ghosts: list of tuples representing ghosts' positions
    :return: the closest distance to Pacman
    """
    ghost_distances = []
    for ghost in ghosts:
        isScared = ghost.scaredTimer > 0
        if not isScared:
            ghost_distances.append(manhattanDistance(ghost.getPosition(),
                                                     pacman))
    return min(ghost_distances) if len(ghost_distances) > 0 else 1

def num_foods(foods):
    return len(foods)

def num_capsules(caps):
    return len(caps)

def closest_scared_ghost(pacman, ghosts):
    """
    Computes the closest distance of any scared ghost to Pacman.

    :param pacman: tuple representing Pacman's position
    :param ghosts: list of tuples representing ghosts' positions
    :return: the closest distance to Pacman
    """
    ghost_distances = []
    for ghost in ghosts:
        isScared = ghost.scaredTimer > 0
        if isScared:
            ghost_distances.append(manhattanDistance(ghost.getPosition(),
                                                     pacman))

    return min(ghost_distances) if len(ghost_distances) > 0 else 1

def closest_capsule(pacman, caps_pos):
    """
    Computes the closest distance of any capsule to Pacman.

    :param pacman: tuple representing Pacman's position
    :param caps_pos: list of tuples representing positions of capsules
    :return: the closest distance to Pacman
    """
    capsule_distances = []
    for caps in caps_pos:
        capsule_distances.append(manhattanDistance(caps, pacman))
    return min(capsule_distances) if len(capsule_distances) > 0 else 9999999





