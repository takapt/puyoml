from collections import namedtuple
from copy import deepcopy

from puyopuyo.game import list_put_actions, Puyo
from puyopuyo.game import Field
from puyopuyo import game


class SimulationResult:
    def __init__(self, field, put_actions=None, max_chains=0, score=0):
        self.field = field
        self.put_actions = put_actions if put_actions else []
        self.max_chains = max_chains
        self.score = score

    def updated_simulation_result(self, put_action, chain_result):
        return SimulationResult(
            self.field,
            [put_action] + self.put_actions,
            max(self.max_chains, chain_result.chains),
            self.score + chain_result.score
        )


def simulate_all_put_actions(field, puyo_pairs):
    assert len(puyo_pairs) <= 3, 'Too many pairs to simulate all put actions in terms of computational cost'

    if len(puyo_pairs) == 0:
        return [SimulationResult(field)]

    simulation_results = []
    puyo_pair = puyo_pairs[0]
    next_puyo_pairs = puyo_pairs[1:]
    for put_action in list_put_actions(field):
        temp_field = deepcopy(field)
        temp_field.put_puyo_pair(puyo_pair, put_action)
        chain_result = temp_field.play_chains()
        if temp_field.is_game_over():
            continue

        next_simulation_results = simulate_all_put_actions(temp_field, next_puyo_pairs)
        for result in next_simulation_results:
            simulation_results.append(result.updated_simulation_result(put_action, chain_result))

    return simulation_results


class ChainsDetail:
    def __init__(self, chains, vanished_at_nth_chain, field):
        self.chains = chains
        self.vanished_at_nth_chain = vanished_at_nth_chain
        self.field = field

    def is_vanished(self, x, y):
        return self.vanished_at_nth_chain[y][x] != -1

    def get_nth_chain_vanish(self, x, y):
        return self.vanished_at_nth_chain[y][x]


def calculate_chains_detail(field_before_play):
    field = deepcopy(field_before_play)
    original_ys = [
        [y if field.get_puyo(x, y) != game.Puyo.EMPTY else -1 for x in range(Field.WIDTH)]
        for y in range(Field.HEIGHT)
    ]
    vanished_at_nth_chain = [[-1 for x in range(Field.WIDTH)] for y in range(Field.HEIGHT)]
    chains = 0
    while True:
        field.drop_puyos()
        chain_result = field.vanish()
        if chain_result['vanished_puyos'] == 0:
            break

        for x, y in chain_result['vanished_pos']:
            assert original_ys[y][x] != -1
            vanished_at_nth_chain[original_ys[y][x]][x] = chains
            original_ys[y][x] = -1

        for x in range(Field.WIDTH):
            top_y = 0
            for y in range(Field.HEIGHT):
                if original_ys[y][x] != -1:
                    ori_puyo = original_ys[y][x]
                    original_ys[y][x] = -1
                    original_ys[top_y][x] = ori_puyo
                    top_y += 1

        chains += 1

    return ChainsDetail(chains, vanished_at_nth_chain, field)


DetectChainsResult = namedtuple('DetectChainsResult', ['x', 'puyo', 'dropped_puyos', 'chains_detail'])


def detect_chains_by_dropping_puyos(field, max_dropped_puyos):
    original_field = deepcopy(field)

    detected_chains_results = []
    for x in range(Field.WIDTH):
        original_height = original_field.get_height(x)
        max_put_y = min(original_height + max_dropped_puyos, Field.HEIGHT)
        for puyo in Puyo.color_puyos():
            field = deepcopy(original_field)
            for y in range(original_height, max_put_y):
                assert field.get_puyo(x, y) == Puyo.EMPTY
                field.set_puyo(x, y, puyo)
                if len(field.list_connected_pos(x, y)) >= Field.VANISH_THRESHOLD:
                    chains_details = calculate_chains_detail(field)
                    detect_chains_result = DetectChainsResult(x, puyo, y - original_height + 1, chains_details)
                    detected_chains_results.append(detect_chains_result)

                    # Should I continue dropping puyos?
                    break

    return detected_chains_results
