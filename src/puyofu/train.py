import itertools
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

import cupy as xp

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib

from ai.util import create_puyo_channels, create_chains_detail_feature
from puyofu.puyofureader import read_puyofu
from puyopuyo import game
from puyopuyo.game import Field, Puyo


class PuyoNet(chainer.Chain):
    def __init__(self):
        super(PuyoNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, out_channels=128, ksize=3, pad=1, nobias=True)
            self.bn1 = L.BatchNormalization(128)

            self.conv2 = L.Convolution2D(None, out_channels=128, ksize=3, pad=1, nobias=True)
            self.bn2 = L.BatchNormalization(128)

            self.conv3 = L.Convolution2D(None, out_channels=128, ksize=3, pad=1, nobias=True)
            self.bn3 = L.BatchNormalization(128)

            # self.conv4 = L.Convolution2D(None, out_channels=128, ksize=3, pad=1, nobias=True)
            # self.bn4 = L.BatchNormalization(128)
            #
            # self.conv5 = L.Convolution2D(None, out_channels=128, ksize=3, pad=1, nobias=True)
            # self.bn5 = L.BatchNormalization(128)

            self.fc1 = L.Linear(None, out_size=512, nobias=True)
            self.bn_fc1 = L.BatchNormalization(512)

            # self.fc2 = L.Linear(None, out_size=32, nobias=True)
            # self.bn_fc2 = L.BatchNormalization(32)

            self.fc3 = L.Linear(None, out_size=1)

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
        #
        # h = self.conv4(h)
        # h = self.bn4(h)
        # h = F.relu(h)
        #
        # h = self.conv5(h)
        # h = self.bn5(h)
        # h = F.relu(h)

        h = F.average_pooling_2d(h, 1)

        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = F.relu(h)
        # h = F.dropout(h, 0.2)

        # h = self.fc2(h)
        # h = self.bn_fc2(h)
        # h = F.relu(h)

        return self.fc3(h)


def create_field_edge():
    edge = np.zeros(shape=(Field.HEIGHT, Field.WIDTH), dtype=np.float32)
    edge[0, :] = edge[:, 0] = edge[:, Field.WIDTH - 1] = 1
    return edge


FIELD_EDGE = create_field_edge()


def create_model_input(field):
    chains_detail_feature = create_chains_detail_feature(field)
    chains_position_arrays = chains_detail_feature.chains_position_numpy_array

    chains = chains_detail_feature.chains_detail.chains
    reversed_order_chains_position_arrays = np.zeros(chains_position_arrays.shape, dtype=np.float32)
    if chains > 0:
        reversed_order_chains_position_arrays[0:chains] = chains_position_arrays[chains-1::-1]

    puyo_channels = create_puyo_channels(field)

    return np.concatenate([
        puyo_channels,
        chains_position_arrays,
        reversed_order_chains_position_arrays,
        FIELD_EDGE.reshape(1, Field.HEIGHT, Field.WIDTH),
    ])


def puyofu_to_model_inputs(puyofu):
    model_inputs = []
    for fields in puyofu:
        for field in fields:
            model_inputs.append(create_model_input(field))
    return np.asarray(model_inputs, dtype=np.float32)


def contains_ojama(field):
    for y in range(game.Field.HEIGHT):
        for x in range(game.Field.WIDTH):
            if field.get_puyo(x, y) == game.Puyo.OJAMA:
                return True
    return False


def is_valid_field(field):
    for x in range(Field.WIDTH):
        for y in range(Field.HEIGHT - 1):
            if field.get_puyo(x, y) == Puyo.EMPTY and field.get_puyo(x, y + 1) != Puyo.EMPTY:
                return False
    return True


def select_fields(puyofu, field_selector):
    selected_puyofu = []
    for fields in puyofu:
        selected_fields = []
        for field in fields:
            if field_selector(field):
                selected_fields.append(field)
        selected_puyofu.append(selected_fields)
    return selected_puyofu


def do_train(puyo_net_version):
    nico_puyofu = read_puyofu('../../data/nico_puyofu.txt')

    gen_puyofu_list = ['../../data/part_weak.txt']
    for i in range(0, puyo_net_version):
        gen_puyofu_list.append('../../data/generator_puyofu_{}.txt'.format(i))
    gen_puyofu = []
    for path in tqdm(gen_puyofu_list, desc='Load gen_puyofu'):
        gen_puyofu.extend(read_puyofu(path))

    # Select valid fields
    nico_puyofu = select_fields(tqdm(nico_puyofu, desc='Select nico_puyofu'),
                                lambda field: not contains_ojama(field) and is_valid_field(field))

    # Select fields which do not appear in nico_puyofu
    nico_field_set = set(game.field_to_one_str(field) for field in itertools.chain.from_iterable(nico_puyofu))
    gen_puyofu = select_fields(gen_puyofu, lambda field: field not in nico_field_set)

    # Split puyofu grouping by game
    nico_train, nico_test = train_test_split(nico_puyofu, test_size=0.1, random_state=puyo_net_version)
    gen_train, gen_test = train_test_split(gen_puyofu, test_size=0.1, random_state=puyo_net_version)

    nico_train = joblib.Parallel(n_jobs=-1)(joblib.delayed(create_model_input)(field) for field in itertools.chain.from_iterable(nico_train))
    nico_test = joblib.Parallel(n_jobs=-1)(joblib.delayed(create_model_input)(field) for field in itertools.chain.from_iterable(nico_test))
    gen_train = joblib.Parallel(n_jobs=-1)(joblib.delayed(create_model_input)(field) for field in itertools.chain.from_iterable(gen_train))
    gen_test = joblib.Parallel(n_jobs=-1)(joblib.delayed(create_model_input)(field) for field in itertools.chain.from_iterable(gen_test))
    # nico_train = puyofu_to_model_inputs(tqdm(nico_train, 'model_inputs for nico_train'))
    # nico_test = puyofu_to_model_inputs(tqdm(nico_test, 'model_inputs for nico_test'))
    # gen_train = puyofu_to_model_inputs(tqdm(gen_train, 'model_inputs for gen_train'))
    # gen_test = puyofu_to_model_inputs(tqdm(gen_test, 'model_inputs for gen_test'))

    def to_chainer_dataset(numpy_fields, labels):
        numpy_fields = xp.asarray(numpy_fields)
        labels = xp.asarray(labels).reshape((len(labels), 1))
        return chainer.datasets.TupleDataset(numpy_fields, labels)

    train = to_chainer_dataset(np.concatenate([nico_train, gen_train]), [1] * len(nico_train) + [0] * len(gen_train))
    test = to_chainer_dataset(np.concatenate([nico_test, gen_test]), [1] * len(nico_test) + [0] * len(gen_test))

    # model
    puyo_net = PuyoNet()
    model = L.Classifier(puyo_net, lossfun=F.sigmoid_cross_entropy, accfun=F.binary_accuracy)
    model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, 128)
    test_iter = chainer.iterators.SerialIterator(test, 128, repeat=False, shuffle=False)

    updater = chainer.training.StandardUpdater(train_iter, optimizer)
    trainer = chainer.training.Trainer(updater, (2, 'epoch'))

    trainer.extend(chainer.training.extensions.Evaluator(test_iter, model))
    trainer.extend(chainer.training.extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.dump_graph('main/loss'))
    trainer.extend(chainer.training.extensions.ProgressBar())

    trainer.run()

    save_filename = 'puyo_net_{}.npz'.format(puyo_net_version)
    chainer.serializers.save_npz(save_filename, puyo_net)
    print('saved: ' + save_filename)


if __name__ == '__main__':
    do_train(0)
