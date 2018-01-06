
from game import Agent
from leftyghost import Leftyghost
from greedyghost import Greedyghost
from algorithm_functions import *

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
        if self.pattern == LEFTY:
            self.strategy = play_lefty
        elif self.pattern == GREEDY:
            self.strategy = play_greedy
        elif self.pattern == SEMI:
            self.strategy = play_semi_random
        else:
            self.pattern = UNKNOWN
            self.strategy = play_unknown

        # Algorithm variables
        self.tree = None
        self.previous = None
        self.num_foods = 0
        self.num_agent = 0
        self.num_scared = 0
        self.ghosts_previous = None
        self.ghosts_proba = None
        self.lefty_proba = None
        self.greedy_proba = None
        self.semi_random_proba = None

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

        self.num_agent = len(state.getGhostPositions()) + 1
        max_depth = self.num_agent * MAX_DEPTH

        # Init the algorithm variables if first turn
        if self.tree is None:
            dist = Counter()
            dist['lefty'], dist['greedy'], dist['semi'] = 1/3, 1/3, 1/3
            self.ghosts_proba = [dist.copy() for _ in range(self.num_agent-1)]
            self.num_foods = num_foods(state.getFood().asList())

        # Perform pattern inference if unknown pattern
        elif self.pattern == UNKNOWN:
            self.pattern_inference(state)

        self.tree = PacmanNode(state)
        self.ghosts_previous = state.getGhostPositions()

        neighbor_tiles, _ = neighbor_lookup(state.getPacmanPosition(),
                                            state.getWalls())
        num_scared = num_scared_ghost(state.getGhostStates())

        # If needed, perform expectiminimax
        if self.previous is None or self.previous == Directions.STOP or\
                len(neighbor_tiles) != 2 or num_scared < self.num_scared:
            v = self.expectimax(self.tree, PACMAN, -math.inf, math.inf,
                                max_depth)

            # Retrieve the best action(s) according to expectiminimax
            actions = []
            for child in self.tree.children:
                if child.score == v:
                    actions.append(child.action)

            action = pick_safe_action(state, deepcopy(actions))

            # If no safe action, only follow expectiminimax for this turn
            if action is not None:
                self.previous = action
            else:
                action = random.choice(actions)
                self.previous = None

        # If engaged in a safe path, just follow it
        else:
            for move in state.getLegalPacmanActions():
                if move != Directions.STOP and \
                                move != Directions.REVERSE[self.previous]:
                    action, self.previous = move, move
                    break

        self.num_scared = num_scared_ghost(state.getGhostStates())

        # If a capsule has been eaten, ghosts are scared and this must be
        # considered
        caps = num_capsules(state.getCapsules())
        caps2 = num_capsules(self.tree.state.getCapsules())
        if caps2 != caps:
            pattern_state = self.tree.state
        else:
            pattern_state = state

        # Compute probability distribution for each pattern and ghost
        if self.pattern == UNKNOWN:
            self.lefty_proba = [play_lefty(pattern_state, i + 1) for i in
                                range(self.num_agent-1)]
            self.greedy_proba = [play_greedy(pattern_state, i + 1) for i in
                                 range(self.num_agent - 1)]
            self.semi_random_proba = [play_semi_random(pattern_state, i + 1)
                                      for i in range(self.num_agent - 1)]

        return action

    def pattern_inference(self, state):
        """
        Tries to infer a still unknown pattern of the ghosts based on their
        previous moves.

        :param state: a GameState object representing the current state of the
                      game.
        """
        ghosts = state.getGhostPositions()
        total_probs = Counter()
        total_probs['lefty'], total_probs['greedy'] = 1, 1
        total_probs['semi'] = 1

        for i in np.arange(len(ghosts)):
            pos, prev = ghosts[i], self.ghosts_previous[i]
            result = np.subtract(pos, prev)
            if result[0] == 0 and 0 < result[1] <= 1:
                action = Directions.NORTH
            elif result[0] == 0 and -1 <= result[1] < 0:
                action = Directions.SOUTH
            elif 0 < result[0] <= 1 and result[1] == 0:
                action = Directions.EAST
            elif -1 <= result[0] < 0 and result[1] == 0:
                action = Directions.WEST
            else:
                action = None

            # Retrieve Probabilities of the strategies for the ghost
            if action is not None:
                lefty = self.lefty_proba[i][action]
                greedy = self.greedy_proba[i][action]
                semi = self.semi_random_proba[i][action]
                self.ghosts_proba[i]['lefty'] *= lefty
                self.ghosts_proba[i]['greedy'] *= greedy
                self.ghosts_proba[i]['semi'] *= semi
                self.ghosts_proba[i].normalize()
                total_probs['lefty'] *= self.ghosts_proba[i]['lefty']
                total_probs['greedy'] *= self.ghosts_proba[i]['greedy']
                total_probs['semi'] *= self.ghosts_proba[i]['semi']
                total_probs.normalize()

        # Retrieve the max probability and check if the threshold is reached
        pattern = total_probs.argMax()
        proba = total_probs[pattern]
        if proba > PATTERN_THRESHOLD:
            if pattern == 'lefty':
                self.strategy = play_lefty
                self.pattern = LEFTY
            elif pattern == 'greedy':
                self.strategy = play_greedy
                self.pattern = GREEDY
            else:
                self.strategy = play_semi_random
                self.pattern = SEMI

    def expectimax(self, node, agent, alpha, beta, depth):
        """
        Performs a max step of the expectiminimax algorithm, with alpha-beta
        pruning.

        :param node: a PacmanNode object representing the current state
        :param agent: a number representing the index of the agent
        :param alpha: a number representing the alpha value of the pruning
        :param beta: a number representing the beta value of the pruning
        :param depth: a number representing the number of turns that can still
                      be played before stopping the search
        :return: the max value of the children of the node
        """

        # If game over or max depth reached
        if node.state.isWin():
            node.set_score(node.state.getScore())
        elif node.state.isLose():
            node.set_score(-math.inf)
        elif depth == 0:
            node.set_score(node.evaluation_function(self.num_foods,
                                                    self.pattern))

        # Search with alpha-beta pruning
        else:
            v = -math.inf

            legal = node.state.getLegalPacmanActions()
            for action in legal:
                state = node.state.generateSuccessor(PACMAN, action)
                child = GhostNode(state, node, action)
                node.children.append(child)

            for child in node.children:
                next_agent = int((agent + 1) % self.num_agent)
                v = max([v, self.expectichance(child, next_agent, alpha, beta,
                                               depth-1)])
                if v >= beta:
                    node.set_score(v)
                    return node.score
                alpha = max(v, alpha)
            node.set_score(v)

        return node.score

    def expectichance(self, node, agent, alpha, beta, depth):
        """
        Performs a chance step of the expectiminimax algorithm, with alpha-beta
        pruning.

        :param node: a GhostNode representing the current state
        :param agent: a number representing the index of the agent
        :param alpha: a number representing the alpha value of the pruning
        :param beta: a number representing the beta value of the pruning
        :param depth: a number representing the number of turns that can still
                      be played before stopping the search
        :return: the expected value of the children of the node
        """

        if node.state.isWin():
            node.set_score(node.state.getScore())
        elif node.state.isLose():
            node.set_score(-math.inf)
        elif depth == 0:
            node.set_score(node.evaluation_function(self.num_foods,
                                                    self.pattern))

        else:
            score = 0
            legal = node.state.getLegalActions(agent)
            probs = self.strategy(node.ghost_state, agent)
            next_agent = int((agent + 1) % self.num_agent)
            for action in legal:
                if probs[action] > PROBA_THRESHOLD:
                    state = node.state.generateSuccessor(agent, action)
                    # Next turn is Pacman
                    if agent == self.num_agent - 1:
                        child = PacmanNode(state, node, action)
                        score += probs[child.action] * \
                            self.expectimax(child, next_agent, alpha, beta,
                                            depth-1)
                    else:
                        child = GhostNode(state, node, action)
                        score += probs[child.action] * \
                            self.expectichance(child, next_agent, alpha, beta,
                                               depth-1)
                    node.children.append(child)

            node.set_score(score)

        return node.score


###########################
#  Expecti Minimax Nodes  #
###########################

class Node:

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.score = 0

    def set_score(self, score):
        self.score = score

    def set_depth(self, depth):
        self.depth = depth

    def evaluation_function(self, start_food, pattern):

        # Get required information
        pacman = self.state.getPacmanPosition()
        ghosts = self.state.getGhostStates()
        foods = self.state.getFood().asList()
        capsules = self.state.getCapsules()
        score = self.state.getScore()
        grid = self.state.getWalls()

        if pattern == LEFTY:
            limit = 0
        elif grid.height*grid.width > 250:
            limit = 2
        else:
            limit = 4

        # Compute eval functions
        next_food = closest_food(pacman, foods, grid)
        _, act_ghost, scared = predict_pos_safeness(pacman, grid, 0, ghosts)
        num_food = num_foods(foods)
        num_caps = num_capsules(capsules)
        num_scared = num_scared_ghost(ghosts)

        if num_scared > 0 and (act_ghost > limit or act_ghost == 0):
            scared *= SCARED_GHOST[HUNT]
            num_scared *= NUM_SCARED[HUNT]
            num_caps *= NUM_CAPS[HUNT]
            act_ghost *= ACT_GHOST[HUNT]
            next_food *= NEXT_FOOD[HUNT]
            num_food *= NUM_FOOD[HUNT]

        elif act_ghost <= limit:
            scared *= SCARED_GHOST[DANGEROUS]
            num_scared *= NUM_SCARED[DANGEROUS]
            num_caps *= NUM_CAPS[DANGEROUS]
            act_ghost *= ACT_GHOST[DANGEROUS]
            next_food *= NEXT_FOOD[DANGEROUS]
            num_food *= NUM_FOOD[DANGEROUS]

        elif limit < act_ghost <= limit*2:
            scared *= SCARED_GHOST[NORMAL]
            num_scared *= NUM_SCARED[NORMAL]
            num_caps *= NUM_CAPS[NORMAL]
            act_ghost *= ACT_GHOST[NORMAL]
            next_food *= NEXT_FOOD[NORMAL]
            num_food *= NUM_FOOD[NORMAL]

        else:
            scared *= SCARED_GHOST[SAFE]
            num_scared *= NUM_SCARED[SAFE]
            num_caps *= NUM_CAPS[SAFE]
            act_ghost *= ACT_GHOST[SAFE]
            next_food *= NEXT_FOOD[SAFE]
            num_food *= NUM_FOOD[SAFE]

        score += next_food + act_ghost + scared + num_food + num_caps + \
            num_scared

        # Return combined score
        return score


class PacmanNode(Node):

    def __init__(self, state, parent=None, action=None):
        super(PacmanNode, self).__init__(state, parent, action)
        self.ghost_state = self.state


class GhostNode(Node):

    def __init__(self, state, parent=None, action=None):
        self.probs = []
        super(GhostNode, self).__init__(state, parent, action)
        self.ghost_state = parent.ghost_state


###############################
#  Ghosts Position Inference  #
###############################

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

    probs.normalize()
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

    probs.normalize()
    return probs


def play_unknown(state, agent):
    """
    Apply all the possible patterns for a ghost for which we don't know the
    pattern at a given state.

    :param state: GameState object representing the current state
    :param agent: number representing the ghost playing
    :return: a Counter object linking legal actions and their probability
    """

    probs = Counter()
    legal = state.getLegalActions(agent)

    # Compute distribution for each possible strategy
    lefty = play_lefty(state, agent)
    greedy = play_greedy(state, agent)
    semi_randy = play_semi_random(state, agent)

    for action in legal:
        probs[action] = (1/3)*lefty[action] + (1/3)*greedy[action] + \
                        (1/3)*semi_randy[action]

    probs.normalize()
    return probs
