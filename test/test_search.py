from unittest import TestCase

from puyopuyo.game import str_rows_to_field, PuyoPair, Puyo
from puyopuyo.search import simulate_all_put_actions


class SimulationAllPutActionsTest(TestCase):
    def test_simulate_all_put_actions(self):
        rows = [
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '  RY  ',
            ' BYRY ',
            ' BYRY ',
            ' BYRY ',
        ]
        field = str_rows_to_field(rows)
        puyo_pairs = [PuyoPair(Puyo.BLUE, Puyo.YELLOW), PuyoPair(Puyo.GREEN, Puyo.GREEN)]

        simulation_results = simulate_all_put_actions(field, puyo_pairs)
        self.assertEqual(484, len(simulation_results))
        self.assertEqual(4, max(result.max_chains for result in simulation_results))
        self.assertEqual(2280, max(result.score for result in simulation_results))
