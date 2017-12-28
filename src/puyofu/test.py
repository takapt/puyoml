import chainer
import chainer.functions as F
import numpy as np

from ai import util
from puyofu import train, puyofureader
from puyopuyo import game

import sys


def print_puyofu_with_prob(puyo_net, puyofu):
    for fields in puyofu:
        for field in fields:
            # print(game.field_to_pretty_str(field))
            game.print_color_field(field)
            numpy_field = util.field_to_numpy_field(field)
            with chainer.using_config('train', False):
                print(F.sigmoid(puyo_net(np.asarray([numpy_field]))))


nico_puyofu = puyofureader.read_puyofu('../../data/part_nico.txt')
nico_puyofu = train.select_fields(nico_puyofu, lambda field: not train.contains_ojama(field))

gen_puyofu = puyofureader.read_puyofu('../../data/generator_puyofu_{}.txt'.format(int(sys.argv[1])))

puyo_net = train.PuyoNet()
chainer.serializers.load_npz('puyo_net_{}.npz'.format(int(sys.argv[1])), puyo_net)
# puyo_net.to_cpu()

print(len(nico_puyofu))
print(len(gen_puyofu))

# with chainer.using_config('train', False):
#     print(F.sigmoid(puyo_net(train.puyofu_to_numpy_fields(nico_puyofu))).data.mean())
#     print(F.sigmoid(puyo_net(train.puyofu_to_numpy_fields(gen_puyofu))).data.mean())

print('gen')
print_puyofu_with_prob(puyo_net, gen_puyofu[:500])

# print('nico')
# print_puyofu_with_prob(puyo_net, nico_puyofu[:50])
