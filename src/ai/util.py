from collections import namedtuple

import numpy as np

from puyopuyo import game, search
from puyopuyo.game import Field


def create_puyo_channels(field):
    puyo_to_channel_index = {
        game.Puyo.EMPTY: 0,
        game.Puyo.RED: 1,
        game.Puyo.BLUE: 2,
        game.Puyo.YELLOW: 3,
        game.Puyo.GREEN: 4,
    }

    np_field = np.zeros((5, Field.HEIGHT, Field.WIDTH), dtype=np.float32)
    for y in range(Field.HEIGHT):
        for x in range(Field.WIDTH):
            puyo = field.get_puyo(x, y)
            np_field[puyo_to_channel_index[puyo], y, x] = 1
    return np_field


ChainsDetailFeature = namedtuple('ChainsDetailFeature', ['chains_detail', 'chains_position_numpy_array'])


def create_chains_detail_feature(field):
    """
    :param field:
    :return: a list of numpy arrays
    """
    detect_chains_results = search.detect_chains_by_dropping_puyos(field, max_dropped_puyos=2)
    best_result = detect_chains_results[np.argmax(result.chains_detail.chains for result in detect_chains_results)]
    chains_detail = best_result.chains_detail

    max_channels = 20
    chains_positions = np.zeros((max_channels, Field.WIDTH, Field.HEIGHT), dtype=np.float32)
    for y in range(Field.HEIGHT):
        for x in range(Field.WIDTH):
            if chains_detail.is_vanished(x, y):
                chains_positions[chains_detail.get_nth_chain_vanish(x, y), x, y] = 1
    return ChainsDetailFeature(chains_positions, chains_detail)
