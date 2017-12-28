import random
from collections import namedtuple

import chainer
import chainer.functions as F
import chainer.links as L
import cupy
import cupy as xp
import numpy as np

from ai.util import create_puyo_channels
from puyopuyo import game, search
from puyopuyo.game import Field


class PuyoNet(chainer.Chain):
    def __init__(self):
        super(PuyoNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, out_channels=256, ksize=3, pad=1, nobias=True)
            self.bn1 = L.BatchNormalization(256)

            self.conv2 = L.Convolution2D(None, out_channels=256, ksize=3, pad=1, nobias=True)
            self.bn2 = L.BatchNormalization(256)

            self.conv3 = L.Convolution2D(None, out_channels=256, ksize=3, pad=1, nobias=True)
            self.bn3 = L.BatchNormalization(256)

            self.conv4 = L.Convolution2D(None, out_channels=256, ksize=3, pad=1, nobias=True)
            self.bn4 = L.BatchNormalization(256)

            self.fc1 = L.Linear(None, out_size=2048, nobias=True)
            self.bn_fc1 = L.BatchNormalization(2048)

            self.fc2 = L.Linear(None, out_size=1)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = F.relu(h)

        h = self.conv3(h)
        h = self.bn3(h)
        h = F.relu(h)

        h = self.conv4(h)
        h = self.bn4(h)
        h = F.relu(h)

        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = F.relu(h)

        return self.fc2(h)


def successive_fields_to_feature(current_field, previous_field):
    return np.concatenate((create_puyo_channels(current_field), create_puyo_channels(previous_field)))


def test():
    s_fields = [
        ((
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '  YY  ',
            ' YRBY ',
            ' BYRBY',
            ' BYRBY',
            ' BYRBY',
        ), 5),
        ((
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '  Y   ',
            ' YRB  ',
            ' RYR  ',
            ' RYR  ',
            ' RYR  ',
        ), 3),
        ((
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '  Y   ',
            ' YRB  ',
            ' RYR  ',
            ' BYR  ',
            ' BYR  ',
        ), 3),
        ((
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '  R   ',
            '  R   ',
            '  YR  ',
            ' YYR  ',
        ), 2),
        ((
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '  YR  ',
            ' YYR  ',
        ), 1),
        ((
             '      ',
             '      ',
             '      ',
             '      ',
             '      ',
             '      ',
             '      ',
             '      ',
             '      ',
             '      ',
             '      ',
             '   R  ',
             '   R  ',
         ), 0)
    ]


    model = PuyoNet()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    inputs = np.array([create_puyo_channels(game.str_rows_to_field(field)) for field, chains in s_fields])
    labels = np.array([[chains] for field, chains in s_fields], dtype=np.float32)

    for _ in range(100):
        predicted = model(inputs)
        loss = F.mean(F.squared_error(predicted, labels))
        print(predicted)
        print(loss)

        model.cleargrads()
        loss.backward()
        optimizer.update()


# def fields_to_numpy_fields(fields):
#     return np.array([field_to_numpy_field(field) for field in fields], dtype=np.float32)


ReplayBufferEntry = namedtuple('ReplayBufferEntry', ['field', 'score'])
EpisodeResult = namedtuple('EpisodeResult', ['field_history', 'chains_history'])


class Agent:
    def __init__(self):
        self.model = PuyoNet()

        if xp == cupy:
            self.model.to_gpu()

        self.optimizer = chainer.optimizers.Adam()
        self.optimizer.setup(self.model)

        self.replay_buffer = []
        self.chains_replay_buffer = []

    def predict_chains(self, features):
        # print(features.shape)
        with chainer.no_backprop_mode():
            return self.model(features)

    def add_episode(self, episode_result):
        ret = 0
        # for field, chains in reversed(list(zip(episode_result.field_history, episode_result.chains_history))):
        for i in range(len(episode_result.field_history) - 1, 0, -1):
            current_field = episode_result.field_history[i]
            previous_field = episode_result.field_history[i - 1]
            chains = episode_result.chains_history[i]

            feature = successive_fields_to_feature(current_field, previous_field)

            ret = 0.95 * ret
            if chains >= 3:
                ret += chains

            self.replay_buffer.append(ReplayBufferEntry(feature, ret))

            if ret >= 2:
                self.chains_replay_buffer.append(ReplayBufferEntry(feature, ret))

            # self.replay_buffer.append(ReplayBufferEntry(field_to_numpy_field(field), ret))
            # if ret > 0:
            #     self.chains_replay_buffer.append(ReplayBufferEntry(field_to_numpy_field(field), ret))

        buffer_size = 2**17
        if len(self.replay_buffer) > buffer_size:
            self.replay_buffer = self.replay_buffer[-buffer_size:]

        if len(self.chains_replay_buffer) > buffer_size:
            self.chains_replay_buffer = self.chains_replay_buffer[-buffer_size:]

    def update_model(self):
        mini_batch = random.sample(self.replay_buffer, min(len(self.replay_buffer), 64))
        mini_batch.extend(random.sample(self.chains_replay_buffer, min(len(self.chains_replay_buffer), 64)))
        x = xp.asarray([field for field, score in mini_batch], dtype=np.float32)
        y = xp.asarray([[score] for field, score in mini_batch], dtype=np.float32)

        predicted_y = self.model(x)
        loss = F.mean(F.squared_error(predicted_y, y))
        print('loss: ', loss)

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()


def generate_episode(agent, epsilon):
    max_turns = 25

    current_field = Field()
    puyo_pair_generator = game.PuyoPairGenerator()
    puyo_pairs = [puyo_pair_generator.generate_puyo_pair() for _ in range(max_turns)]

    # max_chains_in_episode = 0
    field_history = []
    chains_history = []
    for puyo_pair in puyo_pairs:
        results = search.simulate_all_put_actions(current_field, [puyo_pair])
        if len(results) == 0:
            break

        selected_result = None
        if random.random() > epsilon:
            # predicted_chainss = agent.predict_chains([result.field for result in results])
            predicted_chainss = agent.predict_chains(xp.array([
                successive_fields_to_feature(result.field, current_field) for result in results]))

            max_predicted_chains = -1e9
            for result, predicted_chains_after_simulation in zip(results, predicted_chainss):
                if result.field.is_game_over():
                    continue

                predicted_chains = predicted_chains_after_simulation.data
                # predicted_chains = predicted_chains_after_simulation.data
                if predicted_chains > max_predicted_chains:# or result.max_chains >= 2:
                    max_predicted_chains = predicted_chains
                    # if result.max_chains >= 2:
                    #     max_predicted_chains += 2

                    selected_result = result
        else:
            alive_results = [result for result in results if not result.field.is_game_over()]
            if alive_results:
                selected_result = random.choice(alive_results)

        # if selected_result.max_chains > max_chains_in_episode:
        #     max_chains_in_episode = selected_result.max_chains

        if selected_result is None:
            chains_history[-1] = -4
            break

        field_history.append(current_field)
        chains_history.append(selected_result.max_chains)

        current_field = selected_result.field

    return EpisodeResult(field_history, chains_history)


s_test_fields = [
    # 0
    (
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      '
    ),
    # 1
    (
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '   R  ',
        '   R  '
    ),
    # 2
    (
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '   RB ',
        '   RR '
    ),
    # 3
    (
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '    B ',
        '  RRR '
    ),
    # 4
    (
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        'Y     ',
        'Y     ',
        'YR    '
    ),
    # 5
    (
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        ' Y    ',
        ' Y    ',
        'YR    ',
        'YR    ',
        'YR    '
    ),
    # 6
    (
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '  R   ',
        '  R   ',
        '  BR  ',
        'YBBR  '
    ),
    # 7
    (
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '   R  ',
        '   R  ',
        '   BR ',
        ' YBBR '
    ),
    # 8
    (
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        ' Y    ',
        ' YR   ',
        'YRB   ',
        'YRB   ',
        'YRB   '
    ),
    # 8
    (
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '  RYR ',
        '  RYR ',
        '  GRYR',
        ' GGRYR'
    ),
    # 9
    (
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        '      ',
        ' RYR  ',
        ' RYR  ',
        ' GRYR ',
        'GGRYR '
    ),
]
test_fields = [game.str_rows_to_field(s_field) for s_field in s_test_fields]


def main():
    episodes = 10000

    init_epsilon = 0.5
    decay_epsilon = init_epsilon / 5000
    min_epsilon = 0.01

    games = []

    agent = Agent()
    for episode_i in range(episodes):
        epsilon = max(init_epsilon - decay_epsilon * episode_i, min_epsilon)
        episode = generate_episode(agent, epsilon)
        # print(game.field_to_pretty_str(episode.field_history[-1]))

        agent.add_episode(episode)
        print('-' * 50)
        print('# {} {:.4f} {} {}'.format(episode_i, epsilon, len(agent.chains_replay_buffer), max(episode.chains_history)))
        print(game.field_to_pretty_str(episode.field_history[-1]))

        if episode_i > 0 and episode_i % 100 == 0:
            for _ in range(300):
                agent.update_model()

            t = [
                successive_fields_to_feature(field, field)
                for field in test_fields
            ]
            test_predicts = agent.predict_chains(xp.array(t))
            print(test_predicts)

        # if episode_i > 0 and episode_i % 2000 == 0:
        #     chainer.serializers.save_npz('../../models/v1_{}.npz'.format(episode_i), agent.model)

        games.append(episode.field_history)

    def output_puyofu(games):
        with open('weak_ai_puyofu.txt', 'w') as f:
            for fields in games:
                for field in fields:
                    f.write(game.field_to_one_str(field) + '\n')
                f.write('end\n')
    output_puyofu(games)


if __name__ == '__main__':
    main()

# input: Successive fields
# FieldからSub fieldをとって学習に使う