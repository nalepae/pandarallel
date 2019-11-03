import math
import sys

import pytest

from pandarallel.utils import inliner
from types import CodeType, FunctionType

def test_remove_duplicates():
    tuple_ = (3, 1, 2, 2, 1, 4)
    expected_output = (3, 1, 2, 4)

    assert inliner.remove_duplicates(tuple_) == expected_output


def test_int2python_bytes():
    python_version = sys.version_info
    if not (python_version.major == 3 and python_version.minor in (5, 6, 7)):
        with pytest.raises(SystemError):
            inliner.int2python_bytes(4)
        return

    with pytest.raises(OverflowError):
        inliner.int2python_bytes(-1)

    if python_version.minor == 5:
        with pytest.raises(OverflowError):
            inliner.int2python_bytes(65536)

        assert inliner.int2python_bytes(5) == b"\x05\x00"
        assert inliner.int2python_bytes(255) == b"\xFF\x00"
        assert inliner.int2python_bytes(257) == b"\x01\x01"

    else:
        with pytest.raises(OverflowError):
            inliner.int2python_bytes(256)

            assert inliner.int2python_bytes(5) == b"\x05"
        assert inliner.int2python_bytes(255) == b"\xFF"


def test_python_ints2int():
    python_version = sys.version_info
    if not (python_version.major == 3 and python_version.minor in (5, 6, 7)):
        with pytest.raises(SystemError):
            inliner.int2python_bytes(4)
        return

    if python_version.minor == 5:
        with pytest.raises(ValueError):
            inliner.python_ints2int([1, 2, 3])

        with pytest.raises(ValueError):
            inliner.python_ints2int([-1, 2])

        with pytest.raises(ValueError):
            inliner.python_ints2int([1, 256])

        assert inliner.python_ints2int([4, 1]) == 260

    else:
        with pytest.raises(ValueError):
            inliner.python_ints2int([1, 2])

        with pytest.raises(ValueError):
            inliner.python_ints2int([-1])

        with pytest.raises(ValueError):
            inliner.python_ints2int([256])

        assert inliner.python_ints2int([5]) == 5


def test_get_instructions():
    def function(x, y):
        print(x, y)

    python_version = sys.version_info
    if not (python_version.major == 3 and python_version.minor in (5, 6, 7)):
        with pytest.raises(SystemError):
            inliner.get_instructions(function)
        return

    if python_version.minor == 5:
        assert tuple(inliner.get_instructions(function)) == (
            b"t\x00\x00",
            b"|\x00\x00",
            b"|\x01\x00",
            b"\x83\x02\x00",
            b"\x01",
            b"d\x00\x00",
            b"S",
        )
    else:
        assert tuple(inliner.get_instructions(function)) == (
            b"t\x00",
            b"|\x00",
            b"|\x01",
            b"\x83\x02",
            b"\x01\x00",
            b"d\x00",
            b"S\x00",
        )


def test_has_no_return():
    def func_return_nothing(a, b):
        if a > b:
            print(a)
        else:
            print("Hello World!")

    def func_return_something(a, b):
        print(a)
        return b

    def func_several_returns(a, b):
        if a > b:
            print(a)
            return

    python_version = sys.version_info
    if not (python_version.major == 3 and python_version.minor in (5, 6, 7)):
        with pytest.raises(SystemError):
            inliner.has_no_return(func_return_nothing)
        return

    assert inliner.has_no_return(func_return_nothing)
    assert not inliner.has_no_return(func_return_something)
    assert not inliner.has_no_return(func_several_returns)


def test_has_duplicates():
    assert not inliner.has_duplicates([1, 3, 2, 4])
    assert inliner.has_duplicates([1, 3, 2, 3])

def test_get_transitions():
    with pytest.raises(ValueError):
        inliner.get_transitions((1, 2, 2), (1, 2, 3))

    with pytest.raises(ValueError):
        inliner.get_transitions((1, 2), (1, 2, 2))

    olds = ("a", "c", "b", "d")
    news_1 = ("f", "g", "c", "d", "b", "a")
    news_2 = ("c", "d")

    assert inliner.get_transitions(olds, news_1) == {0: 5, 1: 2, 2: 4, 3: 3}
    assert inliner.get_transitions(olds, news_2) == {1: 0, 3: 1}


def test_get_b_transitions():
    transitions = {1: 3, 2: 5, 3: 6}
    byte_source = inliner.OpCode.LOAD_CONST
    byte_dest = inliner.OpCode.STORE_FAST

    python_version = sys.version_info
    if not (python_version.major == 3 and python_version.minor in (5, 6, 7)):
        with pytest.raises(SystemError):
            inliner.get_b_transitions(transitions, byte_source, byte_dest)
        return

    bytes_transitions = inliner.get_b_transitions(transitions, byte_source, byte_dest)

    if python_version.minor == 5:
        expected = {
            (byte_source + b"\x01\x00"): (byte_dest + b"\x03\x00"),
            (byte_source + b"\x02\x00"): (byte_dest + b"\x05\x00"),
            (byte_source + b"\x03\x00"): (byte_dest + b"\x06\x00"),
        }
    else:
        expected = {
            (byte_source + b"\x01"): (byte_dest + b"\x03"),
            (byte_source + b"\x02"): (byte_dest + b"\x05"),
            (byte_source + b"\x03"): (byte_dest + b"\x06"),
        }

    assert bytes_transitions == expected


def test_are_functions_equivalent():
    def a_func(x, y):
        c = 3
        print(c + str(x + y))
        return x * math.sin(y)

    def another_func(x, y):
        c = 4
        print(c + str(x + y))
        return x * math.sin(y)

    assert inliner.are_functions_equivalent(a_func, a_func)
    assert not inliner.are_functions_equivalent(a_func, another_func)


def test_shift_instruction():
    python_version = sys.version_info
    if not (python_version.major == 3 and python_version.minor in (5, 6, 7)):
        with pytest.raises(SystemError):
            inliner.get_b_transitions(transitions, byte_source, byte_dest)
        return

    if python_version.minor == 5:
        assert inliner.shift_instruction(b"d\x05\x00", 3) == b"d\x08\x00"
    else:
        assert inliner.shift_instruction(b"d\x05", 3) == b"d\x08"


def test_shift_instructions():
    python_version = sys.version_info
    if not (python_version.major == 3 and python_version.minor in (5, 6, 7)):
        with pytest.raises(SystemError):
            inliner.get_b_transitions(transitions, byte_source, byte_dest)
        return

    if python_version.minor == 5:
        instructions = (
            b"|\x00\x00",
            b"|\x01\x00",
            b"k\x04\x00",
            # JUMP_POP_IF_FALSE
            b"r\x0f\x00",
            b"n\x00\x00",
            b"|\x00\x00",
            b"|\x01\x00",
            b"k\x04\x00",
            # JUMP_POP_IF_TRUE
            b"s\x1b\x00",
            b"d\x01\x00",
            b"d\x01\x00",
            b"\x04",
            b"\x03",
            b"k\x00\x00",
            # JUMP_IF_FALSE_OR_POP
            b"o2\x00",
            b"d\x01\x00",
            b"k\x02\x00" b"n\x02\x00",
            b"\x02",
            b"\x01",
            b"\x01",
            b"x\x14\x00",
            b"t\x00\x00",
            b"d\x02\x00",
            b"\x83\x01\x00",
            b"D",
            b"]\x06\x00",
            b"}\x02\x00",
            # JUMP_ABSOLUTE
            b"qB\x00",
            b"W",
            b"d\x00\x00",
            b"S",
        )

        expected_shifted_instructions = (
            b"|\x00\x00",
            b"|\x01\x00",
            b"k\x04\x00",
            # JUMP_POP_IF_FALSE
            b"r\x12\x00",
            b"n\x00\x00",
            b"|\x00\x00",
            b"|\x01\x00",
            b"k\x04\x00",
            # JUMP_POP_IF_TRUE
            b"s\x1e\x00",
            b"d\x01\x00",
            b"d\x01\x00",
            b"\x04",
            b"\x03",
            b"k\x00\x00",
            # JUMP_IF_FALSE_OR_POP
            b"o5\x00",
            b"d\x01\x00",
            b"k\x02\x00" b"n\x02\x00",
            b"\x02",
            b"\x01",
            b"\x01",
            b"x\x14\x00",
            b"t\x00\x00",
            b"d\x02\x00",
            b"\x83\x01\x00",
            b"D",
            b"]\x06\x00",
            b"}\x02\x00",
            # JUMP_ABSOLUTE
            b"qE\x00",
            b"W",
            b"d\x00\x00",
            b"S",
        )
    else:
        instructions = (
            b"|\x00",
            b"|\x01",
            b"k\x04",
            # JUMP_POP_IF_FALSE
            b"r\x0f",
            b"n\x00",
            b"|\x00",
            b"|\x01",
            b"k\x04",
            # JUMP_POP_IF_TRUE
            b"s\x1b",
            b"d\x01",
            b"d\x01",
            b"\x04",
            b"\x03",
            b"k\x00",
            # JUMP_IF_FALSE_OR_POP
            b"o2",
            b"d\x01",
            b"k\x02",
            b"n\x02",
            b"\x02",
            b"\x01",
            b"\x01",
            b"x\x14",
            b"t\x00",
            b"d\x02",
            b"\x83\x01",
            b"D\x00",
            b"]\x06",
            b"}\x02",
            # JUMP_ABSOLUTE
            b"qB",
            b"W\x00",
            b"d\x00",
            b"S\x00",
        )

        expected_shifted_instructions = (
            b"|\x00",
            b"|\x01",
            b"k\x04",
            # JUMP_POP_IF_FALSE
            b"r\x12",
            b"n\x00",
            b"|\x00",
            b"|\x01",
            b"k\x04",
            # JUMP_POP_IF_TRUE
            b"s\x1e",
            b"d\x01",
            b"d\x01",
            b"\x04",
            b"\x03",
            b"k\x00",
            # JUMP_IF_FALSE_OR_POP
            b"o5",
            b"d\x01",
            b"k\x02",
            b"n\x02",
            b"\x02",
            b"\x01",
            b"\x01",
            b"x\x14",
            b"t\x00",
            b"d\x02",
            b"\x83\x01",
            b"D\x00",
            b"]\x06",
            b"}\x02",
            # JUMP_ABSOLUTE
            b"qE",
            b"W\x00",
            b"d\x00",
            b"S\x00",
        )

    assert inliner.shift_instructions(instructions, 3) == expected_shifted_instructions


def test_pin_arguments():
    python_version = sys.version_info
    if not (python_version.major == 3 and python_version.minor in (5, 6, 7)):
        with pytest.raises(SystemError):
            inliner.get_b_transitions(transitions, byte_source, byte_dest)
        return

    def func(a, b):
        c = 4
        print(str(a) + str(c))

        return b

    def expected_pinned_func():
        c = 4
        print(str(10) + str(c))

        return 11

    with pytest.raises(TypeError):
        inliner.pin_arguments(func, dict(a=1))

    with pytest.raises(TypeError):
        inliner.pin_arguments(func, dict(a=1, b=2, c=3))

    pinned_func = inliner.pin_arguments(func, dict(a=10, b=11))

    assert inliner.are_functions_equivalent(pinned_func, expected_pinned_func)


def test_inline():
    python_version = sys.version_info
    if not (python_version.major == 3 and python_version.minor in (5, 6, 7)):
        with pytest.raises(SystemError):
            inliner.get_b_transitions(transitions, byte_source, byte_dest)
        return

    def pre_func(b, c):
        a = "hello"
        print(a + " " + b + " " + c)

    def func(x, y):
        try:
            if x > y:
                z = x + 2 * math.sin(y)
                return z ** 2
            elif x == y:
                return 4
            else:
                return 2 ** 3
        except ValueError:
            foo = 0
            for i in range(4):
                foo += i
            return foo
        except TypeError:
            return 42
        else:
            return 33
        finally:
            print("finished")

    def target_inlined_func(x, y):
        # Pinned pre_func
        a = "hello"
        print(a + " " + "pretty" + " " + "world!")

        # func
        try:
            if x > y:
                z = x + 2 * math.sin(y)
                return z ** 2
            elif x == y:
                return 4
            else:
                return 2 ** 3
        except ValueError:
            foo = 0
            for i in range(4):
                foo += i
            return foo
        except TypeError:
            return 42
        else:
            return 33
        finally:
            print("finished")

    inlined_func = inliner.inline(pre_func, func, dict(b="pretty", c="world!"))

    assert inliner.are_functions_equivalent(inlined_func, target_inlined_func)
