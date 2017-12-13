from pacman import Directions
from game import Agent
from util import *
from copy import deepcopy
import numpy as np
import math

class Agentsearch(Agent):
    def __init__(self, index=0, time_eater=40, g_pattern=-1):
        """
        Arguments:
        ----------
        - `index`: index of your agent. Leave it to 0, it has been put
                   only for game engine compliancy
        - `time_eater`: Amount of time pac man remains in `eater`
                        state when eating a big food dot
        - `g_pattern`: Ghosts' pattern in-game. See agentghost.py.
                       Not useful in this class, value does not matter
        """
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
        # Importance given to the heuristic function
        self.h_coefficient = 1
        # Average
        self.average = True

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
        self.goals = state.getFood().asList()
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
        print(start.getGhostPositions())

        self.current = Node(start, None, None, [])
        map = start.getFood()
        dim = sorted([map.height, map.width])
        if dim[1]/dim[0] > 2:
            self.average = False

        nb_goals_left = len(self.goals)

        while len(self.current.goals) > 0:

            # Expand current Node and add children to set of frontier states
            children = self.current.expand_node()
            self.add_open_states(children)

            open_states_costs = np.zeros(len(self.open_states))
            open_states_best = np.zeros(len(self.open_states))

            # Compute the closest frontier state to each goal
            for goal in self.current.goals:
                best, min_cost = self.get_closest_state(goal)
                open_states_best[best] += 1
                open_states_costs[best] += min_cost

            # Average the costs of each frontier state
            for i in np.arange(len(open_states_costs)):
                if open_states_best[i] > 0:
                    open_states_costs[i] = open_states_costs[i] / \
                                           open_states_best[i]

            # Apply progress heuristic to select the best frontier state
            best_state = self.progress_heuristic(open_states_costs,
                                                 open_states_best)

            # Update the state
            self.current = self.open_states[best_state]
            del self.open_states[best_state]

            # If a goal has been reached, clear the frontier states
            if len(self.current.goals) < nb_goals_left:
                self.open_states[:] = []
                self.open_states_positions[:] = []
                nb_goals_left = len(self.current.goals)

        # Retrieve the list of actions to perform
        while self.current.parent != None:
            self.actions.append(self.current.action)
            self.current = self.current.parent

    def add_open_states(self, children):
        """
        Adds new states to the list of frontier states, if it has never
        belonged to the frontier of the current goal search.

        :param children: list of states to add
        """
        for child in children:
            position = child.state.getPacmanPosition()
            if position not in self.open_states_positions:
                self.open_states.append(child)
                self.open_states_positions.append(position)

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
            cost = self.open_states[i].get_heuristic(goal)
            if cost < min_cost:
                best_state = i
                min_cost = cost

        return (best_state, min_cost)

    def progress_heuristic(self, Ds, Gs):
        """
        Apply the progess heuristics method in order to find to best state
        to explore among the frontier set.

        :param Ds: List of the average costs per goal of the frontier states
        :param Gs: List of the number of goals of which the frontier states
                   are the closest
        :return: the best state to explore
        """
        best_state = None
        best_heurisitc = math.inf
        for i in np.arange(len(Ds)):
            if Gs[i] > 0:
                if self.average:
                    heuristic = self.h_coefficient*Ds[i]/Gs[i]
                else:
                    heuristic = self.h_coefficient * Ds[i]
                if heuristic < best_heurisitc:
                    best_state = i
                    best_heurisitc = heuristic

        return best_state

class Node:
    """
    Class representing a Node object designed for a tree search using A*
    algorithm.
    """
    def __init__(self, state, parent, action, path):
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
        # Costs dictionary
        self.costs = dict()
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

    def get_heuristic(self, goal):
        """
        Retrieves the heuristic cost from the state to a given goal.

        :param goal: the goal for which the cost is computed
        :return: the cost to the goal
        """
        if goal in self.goals:
            return self.costs[goal]
        else:
            return 0

    def expand_node(self):
        """
        Expands the node.
        :return: the list of the children of the node
        """
        actions = self.state.getLegalActions(0)

        for action in actions:
            # Does not care about STOP action in search mode
            if (action != Directions.STOP):
                child = self.state.generateSuccessor(0, action)
                position = child.getPacmanPosition()
                # Avoid cycle by checking in the previously visited states
                if position not in self.path:
                    self.children.append(Node(child, self, action,
                                              deepcopy(self.path)))

        return self.children
