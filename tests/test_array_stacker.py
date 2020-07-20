import unittest
from unittest.mock import Mock
from unittest.mock import patch

from utils.array_stacker import ArrayStacker

import numpy as np


class TestArrayStacker(unittest.TestCase):
    def setUp(self):
        self.sut = ArrayStacker()

    def tearDown(self):
        pass

    def test_stack_into_square(self):
        array1 = np.array([["a", "b"], ["c", "d"]])
        array2 = np.array([["e", "f"], ["g", "h"]])
        array3 = np.array([["i", "j"], ["k", "l"]])
        array4 = np.array([["m", "n"], ["o", "p"]])

        result = self.sut.stack_into_square([[array1, array2], [array3, array4]])
        expected_result = np.array(
            [
                ["a", "b", "e", "f"],
                ["c", "d", "g", "h"],
                ["i", "j", "m", "n"],
                ["k", "l", "o", "p"],
            ]
        )

        self.assertTrue(
            np.all(result == expected_result),
            f"""Expected:
# {expected_result}

# Got:
# {result}""",
        )

    def test_stack_row(self):
        array1 = np.array([["a", "b"], ["c", "d"]])
        array2 = np.array([["e", "f"], ["g", "h"]])
        array3 = np.array([["i", "j"], ["k", "l"]])

        result = self.sut.stack_row([array1, array2, array3])

        expected_result = np.array(
            [["a", "b", "e", "f", "i", "j"], ["c", "d", "g", "h", "k", "l"]]
        )

        self.assertTrue(
            np.all(result == expected_result),
            f"""Expected:
# {expected_result}

# Got:
# {result}""",
        )


#         self.assertTrue(
#             np.all(expected_result == result),
#             f"""Expected:
# {expected_result}

# Got:
# {result}""",
#         )


if __name__ == "__main__":
    unittest.main()
