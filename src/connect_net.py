import random
from copy import deepcopy
from collections import deque
import numpy as np
from scipy.signal import convolve2d
from tqdm import tqdm
from multiprocessing import Process, Queue

import keras
from keras.models import Sequential, Input, Model
from keras.layers import Convolution2D, BatchNormalization, Flatten, Dense, Add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K


class Game:
    def __init__(self, n, state=None, width=None, height=None, player=1):
        assert state is not None or (width is not None and height is not None)
        self.n = n
        if state is None:
            self.width = width
            self.height = height
            self.state = np.zeros((self.height, self.width), dtype=np.int)
        else:
            self.state = state
            self.height, self.width = self.state.shape
        self.player = player     # white player = 1, black player = - 1
        self.history = []

    def __str__(self):
        # rows = ['', '%s player\'s turn:' % {-1: 'Black', 1: 'White'}[self.player]]
        rows = []
        for row in self.state:
            rows.append(' '.join([{-1: '●', 0:'.', 1: '○'}[x] for x in row]))
        return '\n'.join(rows)

    def move(self, column):
        if self.state[0, column] != 0:
            return False
        else:
            self.history.append((self.player * self.state, column))     # reverse state for black player
            index = np.where(self.state[:, column] == 0)[0][-1]
            self.state[index, column] = self.player
            self.player *= -1
            return True

    def win(self):
        """Returns list of number of winning combinations for white and black player"""
        wins = [0, 0, 0]
        if not any(self.valid_moves()):
            wins[2] = 1
        kernels = []
        kernels.append(np.ones((1, self.n)))
        kernels.append(np.ones((self.n, 1)))
        kernels.append(np.eye(self.n))
        kernels.append(np.fliplr(np.eye(self.n)))
        for k in kernels:
            response = convolve2d(self.state, k, mode='valid')
            if np.any(response == self.n):
                wins[0] += 1
            if np.any(response == - self.n):
                wins[1] += 1
        return wins

    @staticmethod
    def winner(wins):
        """Returns - 1: black player won, 1: white player won, 0: draw
           Game should be terminated."""
        if wins[0] > wins[1]:
            return 1
        elif wins[0] < wins[1]:
            return - 1
        elif wins[2] == 1:
            return 0
        else:
            raise ValueError('This function should only be called in terminated games, i.e. only one of the players has one or more winning combinations or none of the players have winning combinations and there\'s a draw.')

    def valid_moves(self):
        return self.state[0, :] == 0

    def valid_moves_list(self):
        valid_moves = self.state[0, :] == 0
        return np.arange(self.width)[valid_moves]

    def play_game(self, agent_white, agent_black):
        while True:
            if self.player == 1:
                agent = agent_white
            else:
                agent = agent_black
            move = agent(self)
            self.move(move)
            wins = self.win()
            if sum(wins) > 0:
                if wins[0] > wins[1]:
                    # white wins
                    return 0
                elif wins[0] < wins[1]:
                    # black wins
                    return 1
                else:
                    # draw
                    return 2

    def get_history_batch(self, winner, augment=True):
        """Returns history in batch format. Swaps the states for the black player to the white perspective.
           winner: [0, 1] = [white player won, black player won]
           augment: Boolean: augment data by mirroring
        """
        X_batch = []
        y_batch = []
        for i, (state, column) in enumerate(self.history):
            y = np.zeros(self.width)
            if i % 2 == winner:
                # winner move
                y[column] = 1
            else:
                # loser move
                y[column] = - 1
            X_batch.append(state)
            y_batch.append(y)
            if augment:
                X_batch.append(np.fliplr(state))
                y_batch.append(y[::-1])
        return X_batch, y_batch

    def get_history_q(self, augment=True):
        """Returns history in (state, action, reward, next_state, done) form for Q learning"""
        state_0 = np.zeros((self.height, self.width), dtype=np.int)
        memory = []
        for i in range(len(self.history) - 2):
            state, column = self.history[i]
            next_state, _ = self.history[i + 2]
            action = np.zeros(self.width, dtype=np.int)
            action[column] = 1
            memory.append((state, action, 0, next_state, False))

        # get loser state
        state, column = self.history[-2]
        action = np.zeros(self.width, dtype=np.int)
        action[column] = 1
        memory.append((state, action, - 1, state_0, True))
        # get winner state
        state, column = self.history[-1]
        action = np.zeros(self.width, dtype=np.int)
        action[column] = 1
        memory.append((state, action, 1, state_0, True))

        if augment:
            memory_augmented = []
            for m in memory:
                state, action, reward, next_state, done = m
                memory_augmented.append((np.fliplr(state), action[::-1], reward, np.fliplr(next_state), done))
            memory.extend(memory_augmented)
        return memory


def agent_random(game):
    return np.random.choice(game.valid_moves_list())


def agent_random_s1(state):
    # if there's a winning move, perform it
    state = state.copy()
    valid_moves = state[0, :] == 0
    valid_moves_list = np.arange(len(valid_moves))[valid_moves]
    for move in valid_moves_list:
        game = Game(4, state=state)
        game.move(move)
        wins = game.win()
        if wins[0] > 0:
            return move
    return np.random.choice(valid_moves_list)


def agent_human(game):
    columns = [str(c) for c in range(game.width)]
    print(' '.join(columns))
    print(game)
    move = int(input('Your move: '))
    return move


class RootNode:
    def __init__(self, game, P=0):
        self.game = game
        self.leaf = True
        self.terminal = False
        self.N = 0
        self.W = 0
        self.P = P
        self.v = 0
        self.children = []

    def __str__(self):
        out_str = 'N: %d, W: %d, P: %d' % (self.N, self.W, self.P)
        if self.terminal:
            out_str += ', terminal'
        if self.leaf:
            out_str += ', leaf'
        else:
            out_str += ', %d children' % len(self.children)
        return out_str

    def get_child(self, move):
        child_moves = [child.move for child in self.children]
        try:
            i = child_moves.index(move)
            return self.children[i]
        except ValueError:
            return None

    def visit(self, infer_state_fn, c):
        if self.leaf:
            wins = self.game.win()
            if sum(wins) > 0:
                self.terminal = True
                v = self.game.player * self.game.winner(wins)
            else:
                valid_moves = self.game.valid_moves_list()
                p, v = infer_state_fn(self.game)
                p = p[valid_moves]  # remove invalid moves from prior probabilities
                self.children = [Node(self, move, P) for move, P in zip(valid_moves, p)]
            self.leaf = False
            self.v = v
        elif self.terminal:
            v = self.v
        else:
            max_child = self.get_max_child(c)
            v = max_child.visit(infer_state_fn, c)
        v = - v # switch white and black player
        self.N += 1
        self.W += v
        return v

    def get_max_child(self, c):
        upper_bounds = [(child.Q() + child.U(c)) for child in self.children]
        return self.children[np.argmax(upper_bounds)]


class Node(RootNode):
    def __init__(self, parent, move, P):
        self.parent = parent
        self.move = move
        game = deepcopy(parent.game)
        game.move(move)
        super().__init__(game, P)

    def Q(self):
        return 0 if self.N == 0 else self.W / self.N

    def U(self, c):
        # upper bound function
        return c * self.P * np.sqrt(np.log(self.parent.N) / (1 + self.N))


def random_playout(game):
    num_playouts = 1
    # equal p distribution, is not used here
    p = np.ones(game.width)
    v = 0
    for _ in range(num_playouts):
        g = deepcopy(game)
        end_state = g.play_game(agent_random, agent_random)
        if end_state == 0:
            v += game.player
        elif end_state == 1:
            v -= game.player
    v /= num_playouts
    return p, v


class PolicyGradient:
    """Too simple, doesnt work"""
    def __init__(self, num_games=100, num_iter=10000, width=9, height=6, n=4):
        self.num_games = num_games
        self.num_iter = num_iter
        self.width = width
        self.height = height
        self.n = n
        self.model = self.model_definition()

    def model_definition(self):
        model = Sequential()
        model.add(Convolution2D(16, 3, activation='relu', padding='same', input_shape=(self.height, self.width, 1)))
        model.add(BatchNormalization())
        model.add(Convolution2D(16, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Convolution2D(1, (self.height, 1), activation='relu', padding='valid'))
        model.add(Flatten())
        return model

    def agent(self, state):
        valid_moves = state[0, :] == 0
        p = self.model.predict(state[None, :, :, None])
        p = np.squeeze(p) + 1e-7
        p *= valid_moves
        p /= np.sum(p)
        return np.random.choice(len(p), p=p)

    def train(self):
        self.model.summary()
        adam = Adam(lr=1e-4)
        self.model.compile(loss=policy_loss, optimizer=adam)
        for i in range(self.num_iter):
            game_endings = [0, 0, 0]  # white wins, black wins, draw
            X_batch, y_batch = [], []
            for g in range(self.num_games):
                game = Game(self.width, self.height, self.n)
                winner = game.play_game(self.agent, self.agent)
                game_endings[winner] += 1
                if winner < 2:  # no draw
                    X, y = game.get_history_batch(winner)
                    X_batch.extend(X)
                    y_batch.extend(y)
            if len(X_batch) > 0:
                X = np.stack(X_batch)
                y = np.stack(y_batch)
                self.model.fit(X[..., None], y, batch_size=len(X), epochs=10, verbose=False)

            print(
                '{}/{}: {} games played: {} white wins, {} black wins, {} draws. Batch contains {} states and actions.'.format(
                    i, self.num_iter, self.num_games, *game_endings, len(X_batch)))
            if i % 100 == 0:
                self.validate()

    def validate(self):
        games = []
        moves_list = []
        game_endings = [0, 0, 0]
        for i in range(40):
            game = Game(self.width, self.height, self.n)
            winner = game.play_game(self.agent, agent_random)
            game_endings[winner] += 1
        print('Model (white) vs random (black): {} white wins, {} black wins, {} draws.'.format(*game_endings))
        game_endings = [0, 0, 0]
        for i in range(40):
            game = Game(self.width, self.height, self.n)
            winner = game.play_game(agent_random, self.agent)
            game_endings[winner] += 1
        print('Random (white) vs model (black): {} white wins, {} black wins, {} draws.'.format(*game_endings))
        if self.n == 4:
            moves_list.append([])  # empty board
            moves_list.append([4, 5, 4, 6, 4])  # break the vertical line
            moves_list.append([3, 3, 4, 2, 5])  # break the horizontal line
            moves_list.append([3, 2, 4, 2, 6, 2])  # finish the horizontal line
        if self.n == 3:
            moves_list.append([])
            moves_list.append([1, 0, 2])
            moves_list.append([2, 1, 2])
            moves_list.append([0, 1, 2, 3, 1, 0, 3, 2, 1, 0, 3, 2, 0])
        for moves in moves_list:
            game = Game(width=self.width, height=self.height, n=self.n)
            for move in moves:
                game.move(move)
            p = self.model.predict((game.state * game.player)[None, :, :, None])
            p = np.squeeze(p) + 1e-7
            p /= np.sum(p)
            print(game)
            print(' '.join([str(int(10 * i)) for i in p]))


class DeepQ:
    """Naive, doesnt work"""
    def __init__(self, num_games=1000000, batch_size=64, width=9, height=6, n=4, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.9999):
        self.num_games = num_games
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.n = n
        self.gamma = gamma  # deterministic = 1? I don't think this implementation is deterministic: choice of other player
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.model = self.model_definition()
        self.memory = deque(maxlen=1000)

    def model_definition(self):
        model = Sequential()
        model.add(Convolution2D(64, 3, activation='relu', padding='same', input_shape=(self.height, self.width, 1)))
        model.add(BatchNormalization())
        model.add(Convolution2D(64, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Convolution2D(64, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Convolution2D(64, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Convolution2D(32, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Convolution2D(32, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Convolution2D(32, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Convolution2D(32, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Convolution2D(1, (self.height, 1), activation='linear', padding='valid'))
        model.add(Flatten())
        return model

    def remember(self, memory):
        self.memory.extend(memory)

    def agent(self, state, epsilon=0.):
        # greedy epsilon
        valid_moves = state[0, :] == 0
        if np.random.rand() < epsilon:
            p = valid_moves
            p = p / np.sum(p)
            action = np.random.choice(len(p), p=p)
        else:
            p = self.model.predict(state[None, :, :, None])
            p = np.squeeze(p)
            p *= valid_moves
            action = np.argmax(p)
        return action

    def replay(self):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states_pred = self.model.predict(np.array(states)[..., None])
        next_states_pred = self.model.predict(np.array(next_states)[..., None])
        for state, action, reward, next_state, done, state_pred, next_state_pred in zip(states, actions, rewards, next_states, dones, states_pred, next_states_pred):
            y_target = state_pred
            y_target[action] = reward if done else reward + self.gamma * np.max(next_state_pred)
            x_batch.append(state)
            y_batch.append(y_target)

        self.model.fit(np.array(x_batch)[..., None], np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self):
        self.model.summary()
        adam = Adam(lr=1e-4)
        self.model.compile(loss='mse', optimizer=adam)
        game_endings = [0, 0, 0]  # white wins, black wins, draw
        for i in range(self.num_games):
            game = Game(self.width, self.height, self.n)
            agent = lambda state: self.agent(state, self.epsilon)
            winner = game.play_game(agent, agent)
            game_endings[winner] += 1
            if winner < 2:  # no draw
                memory = game.get_history_q()
                self.remember(memory)

            if i % 100 == 99:
                print('{}/{} 100 last games played: {} white wins, {} black wins, {} draws'.format(i + 1, self.num_games, *game_endings))
                game_endings = [0, 0, 0]
                self.validate()
                print('epsilon: %f' % self.epsilon)

            self.replay()

    def validate(self):
        games = []
        moves_list = []
        game_endings = [0, 0, 0]
        for i in range(20):
            game = Game(self.width, self.height, self.n)
            winner = game.play_game(self.agent, agent_random_s1)
            game_endings[winner] += 1
        print('Model (white) vs random (black): {} white wins, {} black wins, {} draws.'.format(*game_endings))
        game_endings = [0, 0, 0]
        for i in range(20):
            game = Game(self.width, self.height, self.n)
            winner = game.play_game(agent_random_s1, self.agent)
            game_endings[winner] += 1
        print('Random (white) vs model (black): {} white wins, {} black wins, {} draws.'.format(*game_endings))


class DRL:
    def __init__(self, width, height, n, num_filters, num_residual_blocks, batch_size, learning_rate, learning_rate_decay, num_epochs, num_games, memory_size, mcts_budget, mcts_c, mcts_temperature, init_model_path=None):
        self.width = width
        self.height = height
        self.n = n
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.num_epochs = num_epochs
        self.num_games = num_games
        self.memory_size = memory_size
        self.mcts_budget = mcts_budget
        self.mcts_c = mcts_c
        self.mcts_temperature = mcts_temperature
        self.memory = deque(maxlen=self.memory_size)
        if init_model_path is None:
            self.model = self.model_definition(self.width, self.height, num_filters, num_residual_blocks)
        else:
            self.model = keras.models.load_model(init_model_path)

    @staticmethod
    def model_definition(width, height, num_filters, num_residual_blocks):
        # input is 2 layers: stones of both players (1: stone, 0: no), it's always the first player's move
        def residual_block(input):
            conv1 = Convolution2D(num_filters, 3, padding='same', activation='relu')(input)
            norm1 = BatchNormalization()(conv1)
            conv2 = Convolution2D(num_filters, 3, padding='same', activation='relu')(norm1)
            norm2 = BatchNormalization()(conv2)
            output = Add()([norm2, input])
            return output

        input = Input(shape=(height, width, 2))
        conv = Convolution2D(num_filters, 3, padding='same', activation='relu')(input)
        residual_input = BatchNormalization()(conv)
        for _ in range(num_residual_blocks):
            residual_output = residual_block(residual_input)
            residual_input = residual_output

        value_head_conv = Convolution2D(1, 1, padding='same', activation='relu')(residual_input)
        value_head_norm = BatchNormalization()(value_head_conv)
        value_head_flat = Flatten()(value_head_norm)
        value_head_hidden = Dense(height * width, activation='relu')(value_head_flat)
        value_head = Dense(1, activation='tanh', name='value_head')(value_head_hidden)

        policy_head_conv = Convolution2D(2, 1, padding='same', activation='relu')(residual_input)
        policy_head_norm = BatchNormalization()(policy_head_conv)
        policy_head_flat = Flatten()(policy_head_norm)
        policy_head = Dense(width, name='policy_head')(policy_head_flat)

        model = Model(inputs=[input], outputs=[policy_head, value_head])

        return model

    @staticmethod
    def prepare_input(game=None, state=None):
        if state is None:
            state = game.state * game.player
        x = np.stack([state == 1, state == -1], axis=-1).astype(np.int8)
        return x[None, ...]

    def infer_game(self, game):
        x = self.prepare_input(game=game)
        p, v = self.model.predict(x)
        return p.flatten(), v

    def play_game(self, infer_state_fn, budget, c, temperature):
        game = Game(width=self.width, height=self.height, n=self.n)
        root = RootNode(game)
        memory = []
        while True:
            for _ in range(budget):
                root.visit(infer_state_fn, c=c)
            N = np.array([child.N for child in root.children])
            pi_valid = N ** (1 / temperature)
            pi_valid = pi_valid / pi_valid.sum()
            pi = np.zeros(self.width)
            for i, p in enumerate(pi_valid):
                move = root.children[i].move
                pi[move] = p
            pi = pi / pi.sum()
            memory.append((game.state * game.player, pi))
            i = np.random.choice(len(pi_valid), p=pi_valid)
            # new root:
            root = root.children[i]  # remove parent to save memory?
            # make the move
            game.move(root.move)
            wins = game.win()
            if sum(wins) > 0:
                memory.append((game.state * game.player, pi))   # don't forget to append the winning game state..
                break
        if wins[0] > wins[1]:
            print('White player wins.')
            memory_wins = (np.arange(len(memory)) + 1) % 2
        elif wins[0] < wins[1]:
            print('Black player wins.')
            memory_wins = np.arange(len(memory)) % 2
        elif wins[2] == 1:
            print('Draw.')
            memory_wins = np.zeros(len(memory))
        memory_w = []
        for (state, p), w in zip(memory, memory_wins):
            memory_w.append((state, p, w))
        memory_augmented = [(state[:, ::-1], p[::-1], w) for state, p, w in memory_w]
        memory = memory_w + memory_augmented
        return memory

    def fill_memory(self):
        def fill_memory_play(queue):
            while queue.qsize() < self.memory_size:
                memory = self.play_game(random_playout, 10, self.mcts_c, self.mcts_temperature)
                for m in memory:
                    queue.put(m)
                print(queue.qsize())
            print('exitings')
        num_workers = 8
        queue = Queue()
        workers = []
        for _ in range(num_workers):
            p = Process(target=fill_memory_play, args=(queue,))
            workers.append(p)
            p.start()
        while len(self.memory) < self.memory_size:
            m = queue.get()
            self.memory.append(m)
        queue.join()
        for worker in workers:
            worker.join()
            print('join')

    def train(self):
        self.model.compile(
            Adam(lr=self.learning_rate, decay=self.learning_rate_decay),
            loss={'policy_head': 'categorical_crossentropy', 'value_head': 'mean_squared_error'}
        )
        self.model.summary()
        save_checkpoint = ModelCheckpoint('../models/pilot/weights.{epoch:05d}-{loss:.3f}.h5', monitor='loss',
                                          verbose=0, save_best_only=True, save_weights_only=False, mode='auto',
                                          period=1)

        # fill memory first with deep MCTS games
        # self.fill_memory()
        # print('Memory filled with mcts games.')
        while True:
            for i in range(self.num_games):
                memory = self.play_game(self.infer_game, self.mcts_budget, self.mcts_c, self.mcts_temperature)
                self.memory.extend(memory)
                print('%d/%d' % (i, self.num_games))

            self.model.fit_generator(
                self.generate_input(self.batch_size),
                callbacks=[save_checkpoint],
                steps_per_epoch=1,
                epochs=self.num_epochs,
                verbose=2,
                workers=1,
                use_multiprocessing=False
            )

            game = Game(n=4, width=9, height=7)
            mcts_vanilla = MCTS(5000, self.mcts_c)
            mcts_dl = MCTS(5000, self.mcts_c, self.infer_game)
            w = game.play_game(lambda game: mcts_vanilla.agent(game), lambda game: mcts_dl.agent(game))
            print({0: 'MCTS Vanilla won.', 1: 'MCTS DL won!', 2: 'Draw...'}[w])
            print(game)
            game = Game(n=4, width=9, height=7)
            mcts_vanilla = MCTS(5000, self.mcts_c)
            mcts_dl = MCTS(5000, self.mcts_c, self.infer_game)
            w = game.play_game(lambda game: mcts_dl.agent(game), lambda game: mcts_vanilla.agent(game))
            print({0: 'MCTS DL won!', 1: 'MCTS Vanilla won.', 2: 'Draw...'}[w])
            print(game)

    def generate_input(self, batch_size):
        while True:
            mini_batch = random.sample(self.memory, min(len(self.memory), batch_size))
            states, probabilities, wins = zip(*mini_batch)
            x = np.concatenate([self.prepare_input(state=state) for state in states], axis=0)
            p = np.array(probabilities)
            v = np.array(wins)
            yield [x], [p, v]


def random_experiment():
    num_games = 100
    num_columns = 9
    results = [0, 0, 0]     # [white, black, draw]
    for i in range(num_games):
        game = Game()
        winner = game.play_game(agent_random_s1, agent_random)
        results[winner] += 1
    print('{} white wins, {} black wins, {} draws.'.format(*results))


def test_batch_system():
    game = Game()
    moves = [4, 5, 4, 6, 4, 7, 4, 8]
    for m in moves:
        game.move(m)
        wins = game.win()
        if sum(wins) > 0:
            winner = np.argmax(wins)
            X_batch, y_batch = game.get_history_batch(winner)
            break

    for X, y in zip(X_batch, y_batch):
        print(y)
        print(X)


def policy_loss(y_true, y_pred):
    """Custom loss function. Gradient of chosen action should be 1 if good action, -1 if bad action. Other actions 0."""
    return - K.sum(K.tf.multiply(y_true, y_pred))


def mcts():
    budget = 1000
    game = Game(width=7, height=6, n=4)
    root = RootNode(game)
    while True:
        for _ in range(budget):
            root.visit(random_playout, c=1.4)
        scores = [child.N for child in root.children]
        i = np.argmax(scores)
        # new root:
        root = root.children[i] # remove parent to save memory?
        # make the move
        game.move(root.move)
        print(game)
        wins = game.win()
        if wins[0] > wins[1]:
            print('White player wins.')
            break
        elif wins[0] < wins[1]:
            print('Black player wins.')
            break
        elif wins[2] == 1:
            print('Draw.')
            break


class MCTS:
    """Vanilla Monte Carlo Tree Search."""
    def __init__(self, budget, c, infer_state_fn=random_playout):
        self.c = c
        self.budget = budget
        self.infer_state_fn = infer_state_fn
        self.root = None
        self.first_move = True

    def agent(self, game):
        if self.first_move:
            g = deepcopy(game)
            self.root = RootNode(g)
            self.first_move = False
        else:
            opponent_move = game.history[-1][1]
            self.root = self.root.get_child(move=opponent_move)
        for _ in range(self.budget):
            self.root.visit(self.infer_state_fn, c=self.c)
        scores = [child.N for child in self.root.children]
        i = int(np.argmax(scores))
        self.root = self.root.children[i]
        return self.root.move


def play_against_mcts():
    game = Game(n=4, width=9, height=7)
    mcts = MCTS(budget=5000, c=1.4)
    w = game.play_game(lambda game: mcts.agent(game), agent_human)
    print({0: 'The computer won.', 1: 'You won!', 2: 'Draw...'}[w])


if __name__ == '__main__':
    # random_experiment()
    # pg = PolicyGradient(width=7, height=6, n=4)
    # pg.train()
    # dq = DeepQ()
    # dq.train()
    drl = DRL(
        width=9,
        height=7,
        n=4,
        num_filters=256,
        num_residual_blocks=8,
        learning_rate=0.010,
        learning_rate_decay=0.005,
        batch_size=1024,
        num_epochs=100,
        num_games=1000,
        memory_size=1000000,
        mcts_budget=60,
        mcts_c=1.4,
        mcts_temperature=1,
        init_model_path='/home/rembert/connect-net/models/pilot/weights.00095-2.070.h5'
    )
    drl.train()
