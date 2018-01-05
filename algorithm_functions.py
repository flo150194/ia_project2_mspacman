from pacman import Directions
from game import Agent
from util import *
from copy import deepcopy
import numpy as np
import math
from leftyghost import Leftyghost
from greedyghost import Greedyghost

# Evaluation Function Coefficients
NEXT_FOOD = [0, -1, -1.5, -1.75]
ACT_GHOST = [1.5, 2, 1.5, 1]
SCARED_GHOST = [-1.5, -1.5, 0, 0]
NUM_SCARED = [-30, -20, 0, 0]
NUM_FOOD = [-1, -4, -4.5, -4.5]
NUM_CAPS = [20, -50, -50, -50]

# Algorithm Constants
PACMAN = 0
MAX_DEPTH = 1
PATTERN_THRESHOLD = 0.9
PROBA_THRESHOLD = 0.1
LEFTY, GREEDY, SEMI, UNKNOWN = 0, 1, 2, 3
HUNT, DANGEROUS, NORMAL, SAFE = 0, 1, 2, 3


###########################
#  Evaluation Functions  #
###########################

def closest_food(pacman, food_pos, grid):
    """
    Computes the closest distance of any food dot to Pacman.

    :param pacman: tuple representing Pacman's position
    :param food_pos: list of tuples representing positions of food dots
    :param grid: a Grid object representing the maze
    :return: the closest distance to Pacman
    """
    if not food_pos:
        return 0

    seen = [deepcopy(pacman)]

    queue = Queue()
    queue.push((deepcopy(pacman), 0))

    while (not queue.isEmpty()):
        cur = queue.pop()
        if cur[0] in food_pos:
            return cur[1]

        neighbors, actions = neighbor_lookup(cur[0], grid, cur[1])
        for tile in neighbors:
            if tile[0] not in seen:
                queue.push(tile)
                seen.append(tile[0])


def num_foods(foods):
    """
    Computes the remaining amount of food dots.

    :param foods: a list of food coordinates
    :return: the amount of food dots
    """
    return len(foods)


def num_capsules(caps):
    """
    Computes the remaining amount of capsules.

    :param caps: a list of capsules coordinates
    :return: the amount of capsules
    """
    return len(caps)


def num_scared_ghost(ghosts):
    """
    Computes the number of currently scared ghosts

    :param ghosts: a list of AgentStates representing the ghosts
    :return: the number of currently scared ghosts
    """
    nb = 0
    for ghost in ghosts:
        if ghost.scaredTimer > 0:
            nb += 1
    return nb


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
    return min(capsule_distances) if len(capsule_distances) > 0 else 1


#########################
#  Safeness Prediction  #
#########################

def pick_safe_action(state, actions):
    """
    picks a safe action among a list of provided actions.

    :param state: a GameState object representing the current state of the game
    :param actions: a list of actions that can be performed
    :return: a safe action if it exists one, None otherwise
    """

    random.shuffle(actions)
    while actions:
        action = actions.pop()
        if action is not Directions.STOP and \
                predict_path_safeness(state, state.getWalls(),
                                      deepcopy(action)):
            return action

    return None


def predict_path_safeness(state, map, action):
    """
    Predicts the Safeness of a path for Pacman.

    :param state: a GameState object representing the current state of the game
    :param map: a binary grid representing the maze
    :param action: a Direction object representing Pacman's next wished move
    :return: True if the path is safe, False otherwise
    """

    start = state.getPacmanPosition()
    ghosts = state.getGhostStates()

    if action == Directions.NORTH:
        cur = tuple(np.add(start, (0, 1)))
    elif action == Directions.SOUTH:
        cur = tuple(np.add(start, (0, -1)))
    elif action == Directions.EAST:
        cur = tuple(np.add(start, (1, 0)))
    else:
        cur = tuple(np.add(start, (-1, 0)))

    pacman = 1
    while True:
        if cur in state.getCapsules():
            return True
        safe, _, _ = predict_pos_safeness(deepcopy(cur), map, pacman, ghosts)
        if not safe:
            return False
        positions, actions = neighbor_lookup(cur, map)
        if len(positions) > 2:
            break
        for i in range(len(actions)):
            if actions[i] != Directions.REVERSE[action]:
                action = actions[i]
                cur = positions[i][0]
                break
        pacman += 1

    return True


def predict_pos_safeness(pos, grid, pacman_depth, ghosts):
    """
    Predicts the Safeness of a position. A position is safe if Pacman can reach
    it before any ghost assuming the worst case scenario.

    :param pos: a tuple representing the position for which the safeness is
                predicted.
    :param map: a binary grid representing the maze
    :param pacman: a tuple representing Pacman's position
    :param ghosts: a list of tuples representing the ghosts' positions
    :return: True if the position if safe, False otherwise
    """
    ghost_depth = [-1 for x in range(len(ghosts))]
    scared = [-1 for x in range(len(ghosts))]
    seen = [deepcopy(pos)]

    queue = Queue()
    queue.push((deepcopy(pos), 0))
    nb_tiles = grid.height * grid.width - len(grid.asList())

    while (not queue.isEmpty() and len(seen) < nb_tiles):
        cur = queue.pop()
        for i in range(len(ghosts)):
            ghost = ghosts[i]
            cur_pos = cur[0]
            pos = ghost.getPosition()
            if ghost_depth[i] < 0 and scared[i] < 0 and \
                            abs(cur_pos[0] - pos[0]) <= 0.5 and \
                            abs(cur_pos[1] - pos[1]) <= 0.5:
                if ghost.scaredTimer == 0:
                    ghost_depth[i] = cur[1]
                    scared[i] = math.inf
                else:
                    ghost_depth[i] = math.inf
                    scared[i] = cur[1]

        if min(ghost_depth) > 0 and min(scared) > 0:
            break

        neighbors, actions = neighbor_lookup(cur[0], grid, cur[1], 1)
        for tile in neighbors:
            if tile[0] not in seen:
                queue.push(tile)
                seen.append(tile[0])

    if pacman_depth < min(ghost_depth):
        return True, \
               min(ghost_depth) if min(ghost_depth) is not math.inf else 0, \
               min(scared) if min(scared) is not math.inf else 0
    else:
        return False, \
               min(ghost_depth) if min(ghost_depth) is not math.inf else 0, \
               min(scared) if min(scared) is not math.inf else 0


def neighbor_lookup(pos, grid, depth=0, delta=1.0):
    """
    Finds the positions that can be reached in one move from a given position
    in the maze.

    :param pos: a tuple representing the current position
    :param grid: a binary grid representing the maze
    :param delta: number representing the considered displacement
    :return: a list of the reachable positions
    """

    neighbors = []
    actions = []

    x = int(math.ceil(pos[0] + delta))
    y = int(math.ceil(pos[1]))
    if not grid[x][y]:
        neighbors.append((tuple(np.add(pos, (delta, 0))), depth + 1))
        actions.append(Directions.EAST)

    x = int(math.ceil(pos[0] - delta))
    y = int(math.ceil(pos[1]))
    if not grid[x][y]:
        neighbors.append((tuple(np.add(pos, (-delta, 0))), depth + 1))
        actions.append(Directions.WEST)

    x = int(math.ceil(pos[0]))
    y = int(math.ceil(pos[1] + delta))
    if not grid[x][y]:
        neighbors.append((tuple(np.add(pos, (0, delta))), depth + 1))
        actions.append(Directions.NORTH)

    x = int(math.ceil(pos[0]))
    y = int(math.ceil(pos[1] - delta))
    if not grid[x][y]:
        neighbors.append((tuple(np.add(pos, (0, -delta))), depth + 1))
        actions.append(Directions.SOUTH)

    return neighbors, actions