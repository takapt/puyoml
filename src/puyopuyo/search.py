from copy import deepcopy

from puyopuyo.game import list_put_actions


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
