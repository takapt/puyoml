from unittest import TestCase

from puyopuyo.game import Field, Puyo, field_to_str_rows, str_rows_to_field, ChainResult


class FieldStringConvertTest(TestCase):
    def test_field_to_str_rows(self):
        field = Field()
        field.set_puyo(0, 0, Puyo.RED)
        field.set_puyo(1, 0, Puyo.RED)
        field.set_puyo(0, 1, Puyo.BLUE)
        field.set_puyo(3, 3, Puyo.YELLOW)
        field.set_puyo(5, 0, Puyo.YELLOW)
        field.set_puyo(4, 11, Puyo.GREEN)
        field.set_puyo(3, 12, Puyo.GREEN)

        actual = field_to_str_rows(field)

        expected = [
            '   G  ',
            '    G ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '   Y  ',
            '      ',
            'B     ',
            'RR   Y',
        ]
        self.assertListEqual(expected, actual)

    def test_str_rows_to_field(self):
        rows = [
            '   G  ',
            '    G ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '   Y  ',
            '      ',
            'B     ',
            'RR   Y',
        ]

        # :(
        actual = str_rows_to_field(rows)
        restored_str_rows = field_to_str_rows(actual)
        self.assertListEqual(rows, restored_str_rows)


class FieldTest(TestCase):
    def test_list_connected_pos(self):
        rows = [
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '  Y   ',
            '  RB  ',
            ' YYR  ',
            ' BYR  ',
            ' BYR  ',
        ]
        field = str_rows_to_field(rows)

        self.assertSetEqual(set(), field.list_connected_pos(0, 0))
        self.assertSetEqual(set(), field.list_connected_pos(0, 1))
        self.assertSetEqual(set(), field.list_connected_pos(0, 2))
        self.assertSetEqual(set(), field.list_connected_pos(1, 3))
        self.assertSetEqual(set(), field.list_connected_pos(1, 4))

        self.assertSetEqual({(1, 0), (1, 1)}, field.list_connected_pos(1, 0))
        self.assertSetEqual({(1, 0), (1, 1)}, field.list_connected_pos(1, 1))

        self.assertSetEqual({(1, 2), (2, 0), (2, 1), (2, 2)}, field.list_connected_pos(1, 2))

    def test_play_chains_0chains(self):
        rows = [
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            ' B    ',
            ' B    ',
            '  Y   ',
            '  RB  ',
            '  YR  ',
            '  YR  ',
            ' BYR  ',
        ]
        field = str_rows_to_field(rows)
        chain_result = field.play_chains()

        expected_rows = [
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '  Y   ',
            '  RB  ',
            ' BYR  ',
            ' BYR  ',
            ' BYR  ',
        ]
        self.assertListEqual(expected_rows, field_to_str_rows(field))
        self.assertEqual(ChainResult(0, 0), chain_result)

    def test_play_chains_2chains(self):
        rows = [
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            ' Y    ',
            ' B    ',
            '  Y   ',
            '  RB  ',
            '  YR  ',
            '  YR  ',
            ' BYR  ',
        ]
        field = str_rows_to_field(rows)
        chain_result = field.play_chains()

        expected_rows = [
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
            ' B    ',
            ' BYB  ',
        ]
        self.assertListEqual(expected_rows, field_to_str_rows(field))
        self.assertEqual(ChainResult(2, 360), chain_result)

    def test_play_chains_2chains_large_connection(self):
        rows = [
            '      ',
            '      ',
            '      ',
            '  G   ',
            '  G   ',
            '  G   ',
            '  YG  ',
            '  YG  ',
            '  YG  ',
            ' YRYG ',
            ' BRYG ',
            'RBBRG ',
            'RRBRR ',
        ]
        field = str_rows_to_field(rows)
        chain_result = field.play_chains()

        expected_rows = [
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
            '    G ',
            '    G ',
            ' Y  G ',
        ]
        self.assertListEqual(expected_rows, field_to_str_rows(field))
        self.assertEqual(ChainResult(2, 4600), chain_result)

    def test_play_chains_10chains(self):
        rows = [
            '     G',
            '     R',
            '     R',
            '    RB',
            '  BYGY',
            'RGBYGY',
            'RGYYRB',
            'YBBGBB',
            'GBGBRY',
            'GGYGRY',
            'YBYGRY',
            'YYBYGR',
            'BBYGRY',
        ]
        field = str_rows_to_field(rows)
        chain_result = field.play_chains()

        expected_rows = [
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '      ',
            '     G',
            '     R',
            '    RR',
            '    GB',
            'R   GY',
            'R  GRY',
        ]
        self.assertListEqual(expected_rows, field_to_str_rows(field))
        self.assertEqual(ChainResult(10, 39220), chain_result)
