import sys
from puyopuyo import game


def line_to_field(line):
    t = line.split(' ')[0]
    assert len(t) == 6 * 12

    t = t.translate(str.maketrans('012345', ' RBYG*'))
    str_rows = ['      '] + [t[i:i + 6] for i in range(0, 6 * 12, 6)]
    return game.str_rows_to_field(str_rows)


if __name__ == '__main__':
    for line in sys.stdin.readlines():
        if 'end' in line:
            print('end')
            continue

        field = line_to_field(line)
        print(game.field_to_one_str(field))