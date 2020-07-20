from functools import reduce
from typing import List
import numpy as np


class ArrayStacker:
    @classmethod
    def stack_into_square(cls, arrays_to_stack: List[List[np.ndarray]]) -> np.ndarray:
        cls.square_check(arrays_to_stack)

        stacked_rows = [cls.stack_row(row) for row in arrays_to_stack]
        return np.concatenate(stacked_rows, axis=0)

    @staticmethod
    def square_check(arrays_to_stack: List[List[np.ndarray]]):
        number_of_rows = len(arrays_to_stack)

        for row in arrays_to_stack:
            if len(row) != number_of_rows:
                raise RuntimeError("🐳 Number of rows not equal to number of columns")

    @staticmethod
    def stack_row(row_to_stack: List[np.ndarray]):
        return np.concatenate(row_to_stack, axis=1)
