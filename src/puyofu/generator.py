import chainer
import chainer.functions as F
import numpy as np
from tqdm import tqdm

from ai import util
from ai.cnntest import EpisodeResult
from puyofu import train, puyofureader
from puyopuyo import game
import random
from collections import namedtuple

import chainer
import chainer.functions as F
import chainer.links as L
import cupy
import cupy as xp
# import numpy as xp
import numpy as np

from ai.util import create_puyo_channels
from puyopuyo import game, search
from puyopuyo.game import Field
import joblib


class EpisodeGenerator:
    def __init__(self, puyo_net, seed):
        self.puyo_net = puyo_net
        self.puyo_pair_generator = game.PuyoPairGenerator(seed)

    def generate_episode(self):
        max_turns = 30

        current_field = Field()
        puyo_pairs = [self.puyo_pair_generator.generate_puyo_pair() for _ in range(max_turns)]

        field_history = []
        chains_history = []
        for turn, puyo_pair in enumerate(puyo_pairs):
            # with chainer.using_config('train', False):
            #     print(F.sigmoid(self.puyo_net(xp.asarray([util.field_to_numpy_field(current_field)]))))
            # print(game.field_to_pretty_str(current_field))

            results = search.simulate_all_put_actions(current_field, [puyo_pair])
            if len(results) == 0:
                break

            with chainer.using_config('train', False):
                model_inputs = [train.create_model_input(result.field) for result in results]
                human_pred = self.puyo_net(xp.asarray(model_inputs)).data
                human_pred = chainer.cuda.to_cpu(human_pred)

            current_puyos = current_field.count_color_puyos()
            if current_puyos >= 40:
                best_result = results[np.argmax(human_pred)]
            else:
                best_result = None
                highest_pred = -1e60
                for result, pred in zip(results, human_pred):
                    if result.field.count_color_puyos() >= current_puyos + 1 and pred > highest_pred:
                        highest_pred = pred
                        best_result = result

            if best_result is None:
                break

            field_history.append(current_field)
            chains_history.append(best_result.max_chains)

            current_field = best_result.field

        return EpisodeResult(field_history, chains_history)

    # def generate_episode_by_beam_search(self):
    #     max_turns = 25
    #
    #     current_field = Field()
    #     puyo_pairs = [self.puyo_pair_generator.generate_puyo_pair() for _ in range(max_turns)]
    #
    #     field_history = []
    #     chains_history = []
    #     for puyo_pair in puyo_pairs:
    #         print(F.sigmoid(self.puyo_net(xp.asarray([util.field_to_numpy_field(current_field)]))))
    #
    #         results = search.simulate_all_put_actions(current_field, [puyo_pair])
    #         if len(results) == 0:
    #             break
    #
    #         human_pred = self.puyo_net(xp.asarray([util.field_to_numpy_field(result.field) for result in results])).data
    #         human_pred = chainer.cuda.to_cpu(human_pred)
    #         best_result = results[np.argmax(human_pred)]
    #
    #         field_history.append(current_field)
    #         chains_history.append(best_result.max_chains)
    #
    #         current_field = best_result.field
    #
    #     return EpisodeResult(field_history, chains_history)


def output_puyofu(games, path):
    with open(path, 'w') as f:
        for fields in games:
            for field in fields:
                f.write(game.field_to_one_str(field) + '\n')
            f.write('end\n')


def do_generate(puyo_net_version):
    episodes = 200

    puyo_net = train.PuyoNet()
    chainer.serializers.load_npz('puyo_net_{}.npz'.format(puyo_net_version), puyo_net)
    puyo_net.to_gpu()

    generator = EpisodeGenerator(puyo_net, None)

    puyofu = []
    for episode_i in tqdm(range(episodes), total=episodes, desc='generator'):
        episode_result = generator.generate_episode()
        puyofu.append(episode_result.field_history)

    save_path = '../../data/generator_puyofu_{}.txt'.format(puyo_net_version)
    output_puyofu(puyofu, save_path)
    print('saved :', save_path)

    # with chainer.using_config('train', False):
    #     print('adversial pred mean: ', F.sigmoid(puyo_net(xp.asarray(train.puyofu_to_numpy_fields(puyofu)))).data.mean())


if __name__ == '__main__':
    import sys
    puyo_net_version = int(sys.argv[1])
    print('puyo_net_version: {}'.format(puyo_net_version))
    train.do_train(puyo_net_version)
    do_generate(puyo_net_version)

    # for i in range(90, 500):
    #     import gc
    #     gc.collect()
    #     train.do_train(i)
    #     do_generate(i)
