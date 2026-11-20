import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_process.ttnet_data_utils import train_val_data_separation


class GameSplitTest(unittest.TestCase):
    def test_separate_validation_game(self):
        configs = SimpleNamespace(train_game_list=['game_1'], val_games=['game_2'], no_val=False)
        with patch('data_process.ttnet_data_utils.get_events_infor', side_effect=[(['train'], [0]), (['val'], [1])]):
            self.assertEqual(train_val_data_separation(configs), (['train'], ['val'], [0], [1]))

    def test_overlap_is_rejected(self):
        configs = SimpleNamespace(train_game_list=['game_1'], val_games=['game_1'], no_val=False)
        with patch('data_process.ttnet_data_utils.get_events_infor', return_value=([], [])):
            with self.assertRaises(ValueError):
                train_val_data_separation(configs)


if __name__ == '__main__':
    unittest.main()
