from collections import namedtuple
from enum import Enum
from random import Random
from termcolor import colored


class Puyo(Enum):
    EMPTY = 0
    RED = 1
    BLUE = 2
    YELLOW = 3
    GREEN = 4
    OJAMA = 5

    @classmethod
    def color_puyos(cls):
        return [Puyo.RED, Puyo.BLUE, Puyo.YELLOW, Puyo.GREEN]

    def is_color_puyo(self):
        return 1 <= self.value <= 4


ChainResult = namedtuple('ChainResult', ['chains', 'score'])

    
class Field:
    WIDTH = 6
    HEIGHT = 13
    VANISH_THRESHOLD = 4

    DEAD_X, DEAD_Y = 2, 11

    # score calculation: https://www26.atwiki.jp/puyowords/pages/122.html
    CHAIN_BONUS = [0, 0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512]
    COLOR_BONUS = [0, 0, 3, 6, 12, 24]
    CONNECTION_BONUS = [0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 10]

    def __init__(self):
        self.__field = [[Puyo.EMPTY for x in range(Field.WIDTH)] for y in range(Field.HEIGHT)]

    def get_puyo(self, x, y):
        return self.__field[y][x]

    def set_puyo(self, x, y, puyo):
        self.__field[y][x] = puyo

    def get_height(self, x):
        y = 0
        while y < Field.HEIGHT and self.__field[y][x] != Puyo.EMPTY:
            y += 1
        return y

    def put_puyo_pair(self, puyo_pair, put_action):
        self.__field[put_action.get_center_y()][put_action.get_center_x()] = puyo_pair.get_center()
        self.__field[put_action.get_side_y()][put_action.get_side_x()] = puyo_pair.get_side()

    def is_game_over(self):
        return self.__field[Field.DEAD_Y][Field.DEAD_X] != Puyo.EMPTY

    def play_chains(self):
        score = 0
        chains = 0
        while True:
            self.drop_puyos()
            chain_result = self.vanish()
            if chain_result['vanished_puyos'] == 0:
                break

            chains += 1
            score_coef = max(1, Field.CHAIN_BONUS[chains] + chain_result['connection_bonus'] + chain_result['color_bonus'])
            score += 10 * score_coef * chain_result['vanished_puyos']

        return ChainResult(chains, score)

    def drop_puyos(self):
        for x in range(Field.WIDTH):
            top_empty_y = 0
            for y in range(Field.HEIGHT):
                puyo = self.__field[y][x]
                if puyo.is_color_puyo():
                    self.__field[y][x] = Puyo.EMPTY
                    self.__field[top_empty_y][x] = puyo
                    top_empty_y += 1

    def vanish(self):
        vanished_pos = set()
        checked_pos = set()

        connection_bonus = 0
        for y in range(Field.HEIGHT):
            for x in range(Field.WIDTH):
                if (x, y) not in checked_pos and self.__field[y][x].is_color_puyo():
                    connected_pos = self.list_connected_pos(x, y)
                    checked_pos |= connected_pos

                    # The logic to vanish ojama is not implemented
                    if len(connected_pos) >= Field.VANISH_THRESHOLD:
                        vanished_pos |= connected_pos

                        connection_bonus +=\
                            Field.CONNECTION_BONUS[min(len(Field.CONNECTION_BONUS) - 1, len(connected_pos))]

        vanished_colors = set()
        for x, y in vanished_pos:
            vanished_colors.add(self.__field[y][x])
            self.__field[y][x] = Puyo.EMPTY

        return {
            'vanished_pos': vanished_pos,
            'vanished_puyos': len(vanished_pos),
            'connection_bonus': connection_bonus,
            'color_bonus': Field.COLOR_BONUS[len(vanished_colors)]
        }

    def list_connected_pos(self, x, y, connected_positions=None):
        assert Field.is_valid_pos(x, y)

        if connected_positions is None:
            connected_positions = set()

        if not self.__field[y][x].is_color_puyo():
            return connected_positions

        connected_positions.add((x, y))

        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nx, ny = x + dx, y + dy
            if Field.is_valid_pos(nx, ny) and self.__field[y][x] == self.__field[ny][nx] and\
                    (nx, ny) not in connected_positions:
                self.list_connected_pos(nx, ny, connected_positions)

        return connected_positions

    def count_color_puyos(self):
        return sum(1 for y in range(Field.HEIGHT) for x in range(Field.WIDTH) if self.__field[y][x].is_color_puyo())

    @staticmethod
    def is_valid_pos(x, y):
        return (0 <= x < Field.WIDTH) and (0 <= y < Field.HEIGHT)


PUYO_CHAR_PAIRS = [
    (Puyo.EMPTY, ' '),
    (Puyo.RED, 'R'),
    (Puyo.BLUE, 'B'),
    (Puyo.YELLOW, 'Y'),
    (Puyo.GREEN, 'G'),
    (Puyo.OJAMA, '*')
]
PUYO_TO_CHAR = {puyo: c for puyo, c in PUYO_CHAR_PAIRS}
CHAR_TO_PUYO = {c: puyo for puyo, c in PUYO_CHAR_PAIRS}


def field_to_str_rows(field):
    rows = []
    for y in range(Field.HEIGHT - 1, -1, -1):
        rows.append(''.join(PUYO_TO_CHAR[field.get_puyo(x, y)] for x in range(Field.WIDTH)))
    return rows


def field_to_one_str(field):
    return ''.join(field_to_str_rows(field))


def one_str_to_field(one_str):
    return str_rows_to_field([one_str[i:i + Field.WIDTH] for i in range(0, Field.WIDTH * Field.HEIGHT, Field.WIDTH)])


def str_rows_to_field(rows):
    field = Field()
    for y in range(Field.HEIGHT):
        for x in range(Field.WIDTH):
            field.set_puyo(x, y, CHAR_TO_PUYO[rows[Field.HEIGHT - 1 - y][x]])
    return field


def field_to_pretty_str(field):
    return '\n'.join('#' + row + '#' for row in field_to_str_rows(field)) + '\n' + '#' * (1 + Field.WIDTH + 1)


def print_color_field(field):
    for y in range(Field.HEIGHT - 1, -1, -1):
        row = '#'
        for x in range(Field.WIDTH):
            if field.get_puyo(x, y).is_color_puyo():
                row += colored('●', field.get_puyo(x, y).name.lower())
            else:
                row += ' '
        row += '#'
        print(row)
    print('#' * (Field.WIDTH + 2))


class PutAction:
    def __init__(self, center_x, center_y, side_x, side_y):
        self.center_x = center_x
        self.center_y = center_y
        self.side_x = side_x
        self.side_y = side_y

    def get_center_x(self):
        return self.center_x

    def get_center_y(self):
        return self.center_y

    def get_side_x(self):
        return self.side_x

    def get_side_y(self):
        return self.side_y


def list_put_actions(field):
    top_y_by_x = [Field.HEIGHT for x in range(Field.WIDTH)]
    for x in range(Field.WIDTH):
        for y in range(Field.HEIGHT):
            if field.get_puyo(x, y) == Puyo.EMPTY:
                top_y_by_x[x] = y
                break

    put_actions = []
    # vertical
    for x in range(Field.WIDTH):
        if top_y_by_x[x] <= 10:
            y = top_y_by_x[x]
            put_actions.append(PutAction(x, y, x, y + 1))
            put_actions.append(PutAction(x, y + 1, x, y))

    # horizontal
    for x in range(Field.WIDTH - 1):
        if top_y_by_x[x] <= 11 and top_y_by_x[x + 1] <= 11:
            put_actions.append(PutAction(x, top_y_by_x[x], x + 1, top_y_by_x[x + 1]))
            put_actions.append(PutAction(x + 1, top_y_by_x[x + 1], x, top_y_by_x[x]))

    return put_actions


class PuyoPair:
    def __init__(self, center, side):
        self.center = center
        self.side = side

    def get_center(self):
        return self.center

    def get_side(self):
        return self.side


class PuyoPairGenerator:
    def __init__(self, seed=None):
        self.random = Random(seed)

    def generate_puyo_pair(self):
        return PuyoPair(Puyo(self.random.randint(1, 4)), Puyo(self.random.randint(1, 4)))
