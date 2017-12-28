from unittest import TestCase

from puyopuyo import game
from puyopuyo.game import PuyoPair, Puyo, Field
from puyopuyo.search import simulate_all_put_actions, calculate_chains_details


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
        field = game.str_rows_to_field(rows)
        puyo_pairs = [PuyoPair(Puyo.BLUE, Puyo.YELLOW), PuyoPair(Puyo.GREEN, Puyo.GREEN)]

        simulation_results = simulate_all_put_actions(field, puyo_pairs)
        self.assertEqual(484, len(simulation_results))
        self.assertEqual(4, max(result.max_chains for result in simulation_results))
        self.assertEqual(2280, max(result.score for result in simulation_results))


class ChainsDetailsTest(TestCase):
    def test_calculate_chains_details(self):
        rows = [
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            'B     ',
            'Y     ',
            'Y  B  ',
            'GGYB  ',
            'GGRYB ',
            'YBRYB ',
            'YYBRYB',
            'BBRYBB',
        ]
        field = game.str_rows_to_field(rows)
        chains_details = calculate_chains_details(field)

        self.assertEqual(5, chains_details.chains)

        expected_field_rows = [
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '     B',
            '    BB',
        ]
        self.assertListEqual(expected_field_rows, game.field_to_str_rows(chains_details.field))

        expected_nth_chain_vanish = [
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [ 2, -1, -1, -1, -1, -1],
            [ 1, -1, -1, -1, -1, -1],
            [ 1, -1, -1,  4, -1, -1],
            [ 0,  0,  4,  4, -1, -1],
            [ 0,  0,  3,  4,  4, -1],
            [ 1,  2,  3,  4,  4, -1],
            [ 1,  1,  2,  3,  4, -1],
            [ 2,  2,  3,  4, -1, -1],
        ]
        actual_nth_chain_vanish = [
            [chains_details.get_nth_chain_vanish(x, y) for x in range(Field.WIDTH)]
            for y in range(Field.HEIGHT - 1, -1, -1)
        ]

        self.assertListEqual(expected_nth_chain_vanish, actual_nth_chain_vanish)
