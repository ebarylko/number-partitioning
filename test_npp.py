import npp as np

expected = {(1, 1): -20,
            (2, 2): -32,
            (3, 3): -36,
            (1, 2): 16,
            (1, 3): 24,
            (2, 3): 48}


def test_set_qubo():
    assert np.set_qubo({1: 1, 2: 1, 3: 0}, ((1, 1),
                                            (2, 2),
                                            (3, 3),
                                            (1, 2),
                                            (1, 3),
                                            (2, 3))) == expected
