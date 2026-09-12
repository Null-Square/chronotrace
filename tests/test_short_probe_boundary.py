"""Exact two-world ambiguity, even for strongly convex quadratic GD maps."""
from fractions import Fraction as F
from itertools import permutations


def worlds():
    def make(cb, cc):
        return {
            "A": (((F(3, 5), F(7, 100)), (F(7, 100), F(13, 20))), (F(3, 10), F(0))),
            "B": (((F(7, 10), F(1, 10)), (F(1, 10), cb)), (F(3, 10), F(0))),
            "C": (((F(4, 5), F(1, 20)), (F(1, 20), cc)), (F(1, 5), F(0))),
        }
    return make(F(3, 5), F(9, 10)), make(F(17, 30), F(4, 5))


def endpoint(stages, order):
    value = (F(0), F(0))
    for name in order:
        matrix, offset = stages[name]
        value = tuple(sum(a * x for a, x in zip(row, value, strict=True)) + b
                      for row, b in zip(matrix, offset, strict=True))
    return value


def test_identical_short_probes_can_hide_opposite_unique_histories():
    left, right = worlds()
    # Sylvester conditions for 0 < M < I: both M and I-M are positive definite.
    # Therefore each affine map is one GD step on a strongly convex quadratic.
    for world in (left, right):
        for matrix, _ in world.values():
            a, b, c = matrix[0][0], matrix[0][1], matrix[1][1]
            assert matrix[1][0] == b
            assert a > 0 and a * c - b * b > 0
            assert 1 - a > 0 and (1 - a) * (1 - c) - b * b > 0
    for length in (1, 2):
        for probe in permutations("ABC", length):
            assert endpoint(left, probe) == endpoint(right, probe)
    observed = (F(1219, 2000), F(21, 400))
    matches = [[order for order in permutations("ABC") if endpoint(world, order) == observed]
               for world in (left, right)]
    assert matches == [[("A", "B", "C")], [("A", "C", "B")]]
