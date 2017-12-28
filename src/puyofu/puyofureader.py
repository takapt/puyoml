from puyopuyo import game
import sys


def read_puyofu(path):
    with open(path, 'r') as f:
        return read_puyofu_from_file_object(f)


def read_puyofu_from_file_object(f):
    games = []
    fields = []
    for line in f.readlines():
        line = line.strip('\n')
        if line == 'end':
            games.append(fields)
            fields = []
            continue

        assert len(line) == 6 * 13
        fields.append(game.one_str_to_field(line))

    return games


if __name__ == '__main__':
    games = read_puyofu_from_file_object(sys.stdin)
    print(len(games))
    for fields in games:
        for field in fields:
            # print(game.field_to_pretty_str(field))
            game.print_color_field(field)
            print()
        print('-' * 50)