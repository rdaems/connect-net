import numpy as np
from scipy.signal import convolve2d

from keras.models import Sequential
from keras.layers import Convolution2D, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras import backend as K


class Game:
    def __init__(self, width=9, height=6, n=4):
        self.width = width
        self.height = height
        self.n = n
        self.player = 1     # white player = 1, black player = - 1
        self.state = np.zeros((self.height, self.width), dtype=np.int)
        self.history = []

    def __str__(self):
        rows = ['', '%s player\'s turn:' % {-1: 'Black', 1: 'White'}[self.player]]
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
        wins = [0, 0]
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

    def valid_moves(self):
        return self.state[0, :] == 0

    def valid_moves_list(self):
        valid_moves = self.state[0, :] == 0
        return np.arange(self.width)[valid_moves]

    def get_history_batch(self, winner, augment=True):
        """Returns history in batch format. Swaps the states for the black player to the white perspective.
           winner: [0, 1] = [white player won, black player won]
           augment: Boolean: augment data by mirroring
        """
        X_batch = []
        y_batch = []
        history_white = self.history[::2]
        history_black = self.history[1::2]
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


def random_experiment():
    num_games = 10000
    num_columns = 9
    results = {'white': 0, 'black': 0, 'draw': 0}
    for i in range(num_games):
        game = Game()
        while True:
            p = np.random.rand(num_columns)
            p /= np.sum(p)
            p = p * game.valid_moves()
            if np.all(p == 0):
                results['draw'] += 1
                print(game)
                break
            else:
                game.move(np.argmax(p))
                win = game.win()
                if win[0] > 0:
                    results['white'] += 1
                    # print(game)
                    break
                elif win[1] > 0:
                    results['black'] += 1
                    # print(game)
                    break
    print(results)


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


def baseline(num_games, opponent, model, width, height, n):
    if opponent == 'random':
        opponent = lambda game: np.random.choice(game.valid_moves_list())
    game_endings = [0, 0, 0]    # [model, opponent, draw]
    for i in range(num_games):
        game = Game(width, height, n)
        while True:
            opponent_starts = np.random.rand() < 0.5
            if opponent_starts:
                game.move(opponent(game))
            p = model.predict((game.state * game.player)[None, :, :, None])
            p = np.squeeze(p) + 1e-7
            p /= np.sum(p)
            p = p * game.valid_moves()
            if np.all(p == 0):
                game_endings[2] += 1
                break
            else:
                p /= np.sum(p)
                game.move(np.random.choice(width, p=p))
                wins = game.win()
                if sum(wins) > 0:
                    winner = np.argmax(wins)
                    if opponent_starts:
                        game_endings[1 - winner] += 1
                    else:
                        game_endings[winner] += 1
                    break
    return game_endings


def validate(model, width=9, height=6, n=4):
    games = []
    moves_list = []
    game_endings = baseline(100, 'random', model, width=width, height=height, n=n)
    print('{} model wins, {} random wins, {} draws.'.format(*game_endings))
    if n == 4:
        moves_list.append([])   # empty board
        moves_list.append([4, 5, 4, 6, 4])  # break the vertical line
        moves_list.append([4, 4, 5, 3, 6])  # break the horizontal line
        moves_list.append([3, 2, 4, 2, 6, 2])   # finish the horizontal line
    if n == 3:
        moves_list.append([])
        moves_list.append([1, 0, 2])
        moves_list.append([2, 1, 2])
        moves_list.append([0, 1, 2, 3, 1, 0, 3, 2, 1, 0, 3, 2, 0])
    for moves in moves_list:
        game = Game(width=width, height=height, n=n)
        for move in moves:
            game.move(move)
        p = model.predict((game.state * game.player)[None, :, :, None])
        p = np.squeeze(p) + 1e-7
        p /= np.sum(p)
        print(game)
        print(' '.join([str(int(10 * i)) for i in p]))


def main(num_games=100, num_iter=1000, width=9, height=6, n=4):
    model = Sequential()
    model.add(Convolution2D(256, 3, activation='relu', padding='same', input_shape=(height, width, 1)))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(1, (height, 1), activation='relu', padding='valid'))
    model.add(Flatten())

    model.summary()

    adam = Adam(lr=1e-4)
    model.compile(loss=policy_loss, optimizer=adam)

    for i in range(num_iter):
        game_endings = [0, 0, 0]    # white wins, black wins, draw
        X_batch, y_batch = [], []
        for g in range(num_games):
            game = Game(width, height, n)
            while True:
                p = model.predict((game.state * game.player)[None, :, :, None])
                p = np.squeeze(p) + 1e-7
                p /= np.sum(p)
                p = p * game.valid_moves()
                if np.all(p == 0):
                    game_endings[2] += 1
                    break
                else:
                    p /= np.sum(p)
                    game.move(np.random.choice(width, p=p))
                    wins = game.win()
                    if sum(wins) > 0:
                        winner = np.argmax(wins)
                        X, y = game.get_history_batch(winner)
                        X_batch.extend(X)
                        y_batch.extend(y)
                        game_endings[winner] += 1
                        break
        if len(X_batch) > 0:
            X = np.stack(X_batch)
            y = np.stack(y_batch)
            model.fit(X[..., None], y, batch_size=len(X), epochs=2, verbose=False)

        print('{}/{}: {} games played: {} white wins, {} black wins, {} draws. Batch contains {} states and actions.'.format(i, num_iter, num_games, *game_endings, len(X_batch)))
        if i % 1 == 0:
            validate(model, width, height, n)


if __name__ == '__main__':
    main(width=4, height=4, n=3)
