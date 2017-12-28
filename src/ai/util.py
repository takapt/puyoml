import numpy as np

from puyopuyo import game
from puyopuyo.game import Field


def field_to_numpy_field(field):
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
