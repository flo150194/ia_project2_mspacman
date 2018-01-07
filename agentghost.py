from pacman import Directions
from game import Agent
from util import *
from copy import deepcopy
import numpy as np
import math
# XXX: You should complete this class for Step 2


class Agentghost(Agent):
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
        self.current = None
        # Ghosts pattern
        self.pattern = g_pattern
        # Set of frontier states
        self.open_states = []
        # Set of coordinates of all states that have belonged to the frontier
        self.open_states_positions = []
        # Node of the current state of the search
        self.current = None
        # Set of actions that still have to be performed
        self.actions = []
        # Set of goals
        self.goals = []
        # Set of capsules
        self.capsules = []
        # Importance given to the heuristic function
        self.h_coefficient = 1
        # Next goal
        self.next_goal = None

        pass

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
        # Move as long as actions have to performed
        if len(self.actions) > 0:
            action = self.actions.pop()
            return action

        # When no more action, compute the goals and start the search
        pacman = state.getPacmanPosition()
        self.goals = state.getFood().asList()
        self.goals = sorted(self.goals,
                            key=lambda c: manhattanDistance(pacman, c))
        self.capsules = state.getCapsules().asList()
        self.sub_search(state)

        action = self.actions.pop()

        return action

    def sub_search(self, start):
        """
        Performs a search following A* algorithm with progess heuristics
        strategy and Manhattan distance as heuristic function.

        :param start: the start game state of the search as defined
                      in pacman.GameState.
        """

        self.current = DeterministicNode(start, None, None, [], 0)
        map = start.getFood()
        dim = sorted([map.height, map.width])

        self.next_goal = self.goals.pop()[-1]
        while len(self.goals) > 0:

            # Expand current Node and add children to set of frontier states
            self.current.expand_node()
            if self.pattern == 0:
                children = self.current.play_leftyghost()
            if self.pattern < 1:
                self.add_deterministic_open_states(children)

            # Apply heuristic to select the best frontier state
            best_state = self.get_closest_state(self.next_goal)

            # Update the state
            self.current = self.open_states[best_state]
            del self.open_states[best_state]

            self.next_strategy()

        # Retrieve the list of actions to perform
        while self.current.parent is not None:
            self.actions.append(self.current.action)
            self.current = self.current.parent

    def next_strategy(self):
        """
        Update pacman's strategy based on the current state.
        """

        # If pacman is in feeding mode
        if self.current.strategy == 0:
            # If a goal has been reached, clear the frontier states and select
            # a new goal.
            distance = self.current.get_feeding_heuristic(self.next_goal)
            if distance == 0:
                self.open_states[:] = []
                self.open_states_positions[:] = []
                pacman = self.current.state.getPacmanPosition()
                self.goals = sorted(self.goals,
                                    key=lambda c: manhattanDistance(pacman, c),
                                    reverse=True)
                self.next_goal = self.goals.pop()

        # If pacman is in capsule mode
        elif self.current.strategy == 1:
            # Do Something
            m = 1
        # If pacman is in hunting mode
        else:
            m = 1

    def add_deterministic_open_states(self, children):
        """
        Adds new states to the list of frontier states, if it has never
        belonged to the frontier of the current goal search and if no ghost
        has reached Pacman.

        :param children: list of states to add
        """
        for child in children:
            position = child.state.getPacmanPosition()
            ghosts = child.ghosts
            dist = self.closest_ghost(position, ghosts)
            if (position not in self.open_states_positions) and dist > 0:
                self.open_states.append(child)
                self.open_states_positions.append(position)

    @staticmethod
    def closest_ghost(pacman, ghosts):
        """
        Compute the shortest distance from Pacman to any ghost.

        :param pacman: Pacman's position
        :param ghosts: List of ghosts position
        :return: the minimum manhattan distance from Pacamn to any ghost
        """
        min_dist = math.inf
        for g in ghosts:
            dist = manhattanDistance(pacman, g)
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def get_closest_state(self, goal):
        """
        Retrieves the closest state to a given goal and its cost. If the goal
        has already been reached before this state, then the cost of this state
        is equal to zero.

        :param goal: pair of coordinates of the goal
        :return: (best_state, min_cost): the pair clostest state/cost
        """
        best_state = None
        min_cost = math.inf
        for i in np.arange(len(self.open_states)):
            cost = self.open_states[i].get_feeding_heuristic(goal)
            if cost < min_cost:
                best_state = i
                min_cost = cost

        return best_state

    def get_closest_capsule(self):
        """
        Compute the closest capsule from the current pacman's position

        :return: the closest capsule from pacman
        """
        caps = None
        min_cost = math.inf
        capsules = self.current.capsules
        for i in np.arange(len(capsules)):
            cost = self.current.get_capsules_heuristic(capsules[i])
            if cost < min_cost:
                caps = capsules[i]
                min_cost = cost

        return caps, min_cost


class Node:
    """
        Class representing a Node object designed for a tree search using A*
        algorithm.
        """

    def __init__(self, state, parent, action, path, strategy):
        # Set of children states, empty while not expanded
        self.children = []
        # Parent of the node
        self.parent = parent
        # Action (as definded in class pacman.py) that lead to the node
        self.action = action
        # GameState of the node
        self.state = state
        # Goals that have not been reached yet
        self.goals = state.getFood().asList()
        # Capsules
        self.capsules = state.getCapsules().asList()
        # Costs dictionary
        self.costs = dict()
        # Costs dictionary for Capsules
        self.costs_capsules = dict()
        # Current strategy of Pacman
        self.strategy = strategy
        # Back_cost is the backward cost of the state
        # path is the set of previously visited states for a given goals set
        if parent is not None:
            self.back_cost = parent.back_cost + 1
            if len(self.goals) < len(parent.goals):
                self.path = [state.getPacmanPosition()]
            else:
                self.path = path
                self.path.append(state.getPacmanPosition())
        else:
            self.back_cost = 0
            self.path = [state.getPacmanPosition()]

        # Fill costs dictionary for remaining goals
        for i in self.goals:
            cost = self.back_cost + \
                   manhattanDistance(self.state.getPacmanPosition(), i)
            self.costs[i] = cost
        for i in self.capsules:
            cost = self.back_cost + \
                   manhattanDistance(self.state.getPacmanPosition(), i)
            self.costs_capsules[i] = cost


class DeterministicNode(Node):
    """
    Class representing a Node object designed for a tree search using A*
    algorithm.
    """
    def __init__(self, state, parent, action, path, strategy):
        # Ghosts position
        self.ghosts = self.state.getGhostPositions()

        super(DeterministicNode, self).__init__(state, parent, action, path,
                                                strategy)

    def get_feeding_heuristic(self, goal):
        """
        Retrieves the heuristic cost from the state to a given goal.

        :param goal: the goal for which the cost is computed
        :return: the cost to the goal
        """
        if goal in self.goals:
            return self.costs[goal]
        else:
            return 0

    def get_capsules_heuristic(self, caps):
        if caps in self.capsules:
            return self.costs_capsules[caps]
        else:
            return 0

    def expand_node(self):
        """
        Expands the node.
        """
        actions = self.state.getLegalActions(0)

        for action in actions:
            # Does not care about STOP action in search mode
            if action != Directions.STOP:
                child = self.state.generateSuccessor(0, action)
                position = child.getPacmanPosition()
                # Avoid cycle by checking in the previously visited states
                if position not in self.path:
                    node = DeterministicNode(child, self, action,
                                             deepcopy(self.path))
                    self.children.append(node)

    @staticmethod
    def get_next_move_leftypattern(legal):
        """
        Determines the next action to perform for a ghost following the
        lefty pattern.

        :param legal: List of legal actions
        :return: the next action to perform
        """
        left = Directions.WEST
        if left in legal:
            return left
        south = Directions.SOUTH
        if south in legal:
            return south
        east = Directions.EAST
        if east in legal:
            return east
        north = Directions.NORTH
        if north in legal:
            return north

    def play_leftyghost(self):
        """
        Make move the ghosts following the lefty pattern on all the children
        of the node.

        :return: the list of the children of the node
        """
        for child in self.children:
            for k in np.arange(len(self.ghosts)):
                legal = child.state.getLegalActions(k+1)
                action = self.get_next_move_leftypattern(legal)
                if not (child.state.isWin() or child.state.isLose()):
                    child.state = child.state.generateSuccessor(k+1, action)
                else:
                    break

            child.ghosts = child.state.getGhostPositions()

        return self.children


class StochasticNode(Node):

    def __init__(self, state, parent, action, path, strategy, probs=None):

        # Create the probas grid if root of the tree
        if probs is None:
            ghosts = state.getGhostPositions()
            nb_ghosts = len(ghosts)
            height = state.getWalls().height
            width = state.getWalls().width
            self.probs = np.zeros((height, width, nb_ghosts))
            for i in np.arange(len(ghosts)):
                coord = ghosts[i][::-1]
                coord += (i,)
                probs[coord] = 1.0
        else:
            self.probs = probs

        super(StochasticNode, self).__init__(state, parent,
                                             action, path, strategy)

    @staticmethod
    def update_prob_greedy(probs, walls, pacman_pos):
        """Update a probability grid according to a greedy ghost strategy.

        :param probs: a numpy array of floats representing the initial
        probabilities in the grid.
        :param walls: a numpy array of booleans representing the walls in the
        grid.
        :param pacman_pos: a pair of indices representing Pacman's position in
        the grid.
        :return: a numpy array of floats which is the result of a one step
        update of the initial probability grid according to a greedy ghost
        strategy.
        """
        old_probs = probs.copy()
        width = walls.width
        height = walls.height

        # Get all non-null elements in the probability grid
        rows, cols = probs.nonzero()

        for i, j in zip(rows, cols):
            valid_positions = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
            valid_positions = list(filter(lambda x: (0 <= x[0] < width) and
                                                    (0 <= x[1] < height) and
                                          not walls[x[0]][x[1]],
                                          valid_positions))

            # Find min distance positions
            min_pos = min(valid_positions, key=lambda x:
                          manhattanDistance(x, pacman_pos))
            min_dist = manhattanDistance(min_pos, pacman_pos)
            possible_pos = [pos for pos in valid_positions
                            if manhattanDistance(pos, pacman_pos) == min_dist]

            # Update the grid
            n_pos = len(possible_pos)
            for row, col in possible_pos:
                probs[row, col] += float(old_probs[i, j]) / float(n_pos)
            if n_pos > 0:
                probs[i, j] -= old_probs[i, j]

        return probs

    @staticmethod
    def update_prob_lefty(probs, walls, direction):
        """Update a probability grid according to a counterclockwise left
        strategy.

        :param probs: a numpy array of floats representing the initial
        probabilities in the grid.
        :param walls: a numpy array of booleans representing the walls in the
        grid.
        :param direction: a type of Directions to update the ghost's
        probability.
        :return: a numpy array of floats which is the result of a one step
        update of the initial probability grid according to a counterclockwise
        left strategy.

        Rem :
        -----
        This method does not work as is and requires knowing the direction of
        ghosts multiple steps in advance for it to behave properly.
        """
        old_probs = probs.copy()
        width = walls.width
        height = walls.height

        # Get all non-null elements in the probability grid
        rows, cols = probs.nonzero()

        for i, j in zip(rows, cols):
            blocked = False

            if direction != Directions.STOP:
                if direction == Directions.LEFT and 0 <= i - 1 \
                        and not walls[i - 1][j]:
                    valid_positions = [(i - 1, j)]
                elif direction == Directions.SOUTH and 0 <= j - 1 \
                        and not walls[i][j - 1]:
                    valid_positions = [(i, j - 1)]
                elif direction == Directions.EAST and i + 1 < width \
                        and not walls[i + 1][j]:
                    valid_positions = [(i + 1, j)]
                elif direction == Directions.NORTH and j + 1 < height \
                        and not walls[i][j + 1]:
                    valid_positions = [(i, j + 1)]
                else:
                    blocked = True
            if blocked:
                # The list is ordered in a counterclockwise fashion starting
                # from the WEST direction
                valid_positions = [(i - 1, j), (i, j - 1),
                                   (i + 1, j), (i, j + 1)]
                valid_positions = list(filter(lambda x: (0 <= x[0] < width) and
                                              (0 <= x[1] < height) and
                                              not walls[x[0]][x[1]],
                                              valid_positions))

            # Only update the probabilities if the ghost can move
            if len(valid_positions) > 0:
                row, col = valid_positions[0]
                probs[row, col] += old_probs[i, j]
                probs[i, j] -= old_probs[i, j]

        return probs

    @staticmethod
    def update_prob_random_valid(probs, walls):
        """Update a probability grid according to a random valid move strategy.

        :param probs: a numpy array of floats representing the initial
        probabilities in the grid.
        :param walls: a numpy array of booleans representing the walls in the
        grid.
        :return: a numpy array of floats which is the result of a one step
        update of the initial probability grid according to a random valid move
        strategy.
        """
        old_probs = probs.copy()
        width = walls.width
        height = walls.height

        # Get all non-null elements in the probability grid
        rows, cols = probs.nonzero()

        for i, j in zip(rows, cols):
            valid_positions = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
            valid_positions = list(filter(lambda x: (0 <= x[0] < width) and
                                                    (0 <= x[1] < height) and
                                          not walls[x[0]][x[1]],
                                          valid_positions))

            n_pos = len(valid_positions)
            for row, col in valid_positions:
                probs[row, col] += float(old_probs[i, j]) / float(n_pos)
            if n_pos > 0:
                probs[i, j] -= old_probs[i, j]

        return probs

    def play_greedy(self):
        """
        Updates the probabilities grid of the ghosts following the greedy
        pattern.
        """
        # Get Pacman's current position
        pacman_pos = self.state.getPacmanPosition()
        walls = self.state.getWalls()

        # Get the number of ghosts in the grid
        _, _, num_ghosts = self.probs.shape

        # Update the probability grid for each ghost
        for ghost in range(num_ghosts):
            self.probs[:, :, ghost] = \
                self.update_prob_greedy(self.probs[:, :, ghost], walls,
                                        pacman_pos)

    def play_semi_random(self):
        """
        Updates the probabilities grid of the ghosts following the semi-random
        pattern.
        """
        pass

    def play_unknown(self):
        """
        Updates the probabilities grid of the ghosts following the unknown
        pattern.
        """
        pass
