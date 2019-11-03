import dis
import re
import sys
from inspect import signature
from itertools import chain, tee
from types import CodeType, FunctionType
from typing import Any, Dict, Iterable, List, Tuple


class OpCode:
    JUMP_ABSOLUTE = b"q"
    JUMP_IF_FALSE_OR_POP = b"o"
    JUMP_IF_TRUE_OR_POP = b"p"
    LOAD_ATTR = b"j"
    LOAD_CONST = b"d"
    LOAD_FAST = b"|"
    LOAD_GLOBAL = b"t"
    LOAD_METHOD = b"\xa0"
    POP_JUMP_IF_FALSE = b"r"
    POP_JUMP_IF_TRUE = b"s"
    RETURN_VALUE = b"S"
    STORE_ATTR = b"_"
    STORE_FAST = b"}"


def ensure_python_version(function):
    """Raise SystemError if Python version not in 3.{5, 6, 7}"""

    def wrapper(*args, **kwargs):
        python_version = sys.version_info
        if not (python_version.major == 3 and python_version.minor in (5, 6, 7)):
            raise SystemError("Python version should be 3.{5, 6, 7}")

        return function(*args, **kwargs)

    return wrapper


def remove_duplicates(tuple_: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Remove duplicate in tuple `tuple_`.

    Example: tuple_ = (3, 1, 2, 2, 1, 4)
             The returned tuple is: (3, 1, 2, 4)
    """

    return tuple(sorted(set(tuple_), key=tuple_.index))


@ensure_python_version
def int2python_bytes(item: int) -> bytes:
    """Convert an integer into Python bytes, depending of the Python version.

    Examples:

    With Python 3.5:
    int2python_bytes(5) = b"\x05\x00"
    int2python_bytes(255) = b"\xFF\x00"
    int2python_bytes(257) = b"\x01\x01"

    With Python 3.{6, 7}:
    int2python_bytes(5) = b"\x05"
    int2python_bytes(255) = b"\xFF"

    If Python version is not 3.5, 3.6 or 3.7, a SystemError is raised.
    For Python 3.5, if item not in [0, 65535], a OverflowError: is raised.
    For Python 3.{6, 7}, if item not in [0, 255], a OverflowError: is raised.
    """
    python_version = sys.version_info

    nb_bytes = 2 if python_version.minor == 5 else 1
    return int.to_bytes(item, nb_bytes, "little")


@ensure_python_version
def python_ints2int(items: List[int]) -> int:
    """Convert Python integer (depending of Python version) to integer

    If Python 3.5:
        python_ints2int([3, 2]) = 3 + 2 * 256 = 515
        python_ints2int([4, 0]) = 4 + 0 * 256 = 4

    If Python 3.{6, 7}:
        python_ints2int([3]) = 3
        python_ints2int([0]) = 0

    If Python version is not 3.5, 3.6 or 3.7, a SystemError is raised.
    If at least one element of items is not in [0, 255], a ValueError is raised.

    If Python version is 3.5 and items does not contain exactly 2 elements, a ValueError
    is raised.

    If Python version is 3.{6, 7} and items does not contain exactly 1 element, a
    ValueError is raised.
    """
    if not all(0 <= item <= 255 for item in items):
        raise ValueError("Each element of items shoud be in [0, 255]")

    python_version = sys.version_info
    if python_version.minor == 5:
        if len(items) != 2:
            raise ValueError("items should contain exactly 2 elements")
        return items[0] + (items[1] << 8)
    else:
        if len(items) != 1:
            raise ValueError("items should contain exactly 1 elements")
        return items[0]


@ensure_python_version
def get_instructions(func: FunctionType) -> Iterable[bytes]:
    """Return a list of bytes where each item of a list correspond to an instruction.

    Exemple:
    def function(x, y):
        print(x, y)

    With Python 3.5, corresponding pretty bytecode is:
    1           0 LOAD_GLOBAL              0 (print)
                3 LOAD_FAST                0 (x)
                6 LOAD_FAST                1 (y)
                9 CALL_FUNCTION            2 (2 positional, 0 keyword pair)
               12 POP_TOP
               13 LOAD_CONST               0 (None)
               16 RETURN_VALUE

    Corresponding bytecode is: b't\x00\x00|\x00\x00|\x01\x00\x83\x02\x00\x01d\x00\x00S'

    tuple(get_instructions(function)) = (b't\x00\x00', b'|\x00\x00',
                                         b'|\x01\x00', b'\x83\x02\x00', b'\x01',
                                         b'd\x00\x00', b'S')

    With Python 3.6 & 3.7, corresponding bytecode is:
    1           0 LOAD_GLOBAL              0 (print)
                2 LOAD_FAST                0 (x)
                4 LOAD_FAST                1 (y)
                6 CALL_FUNCTION            2
                8 POP_TOP
               10 LOAD_CONST               0 (None)
               12 RETURN_VALUE

    Corresponding bytecode is: b't\x00|\x00|\x01\x83\x02\x01\x00d\x00S\x00'

    tuple(get_instructions(function)) = (b't\x00', b'|\x00', b'|\x01',
                                         b'\x83\x02', b'\x01\x00', b'd\x00',
                                         b'S\x00')

    If Python version not in 3.{5, 6, 7}, a SystemError is raised.
    """

    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    func_co_code = func.__code__.co_code
    len_bytecode = len(func_co_code)

    instructions_offsets = tuple(instr.offset for instr in dis.Bytecode(func)) + (
        len_bytecode,
    )

    return (func_co_code[start:stop] for start, stop in pairwise(instructions_offsets))


@ensure_python_version
def has_no_return(func: FunctionType) -> bool:
    """Return True if `func` returns nothing, else return False

    If Python version not in 3.{5, 6, 7}, a SystemError is raised.
    """

    code = func.__code__

    co_code = code.co_code
    co_consts = code.co_consts

    instructions = tuple(get_instructions(func))

    return_offsets = tuple(
        index
        for index, instruction in enumerate(instructions)
        if instruction[0] == int.from_bytes(OpCode.RETURN_VALUE, byteorder="little")
    )

    load_const_none = OpCode.LOAD_CONST + bytes((co_consts.index(None),))

    return len(return_offsets) == 1 and instructions[-2][0:2] == load_const_none


def has_duplicates(tuple_: Tuple):
    """Return True if `tuple_` contains duplicates items.

    Exemple: has_duplicates((1, 3, 2, 4)) == False
             has_duplicates((1, 3, 2, 3)) == True
    """

    return len(set(tuple_)) != len(tuple_)


def get_transitions(olds: Tuple, news: Tuple) -> Dict[int, int]:
    """Returns a dictionnary where a key represents a position of an item in olds and
    a value represents the position of the same item in news.

    If an element of `olds` is not in `news`, then the corresponding value will be
    `None`.

    Exemples:
    olds = ("a", "c", "b", "d")
    news_1 = ("f", "g", "c", "d", "b", "a")
    news_2 = ("c", "d")

    get_transitions(olds, news_1) = {0: 5, 1: 2, 2: 4, 3: 3}
    get_transitions(olds, news_2) = {1: 0, 3: 1}

    `olds` and `news` should not have any duplicates, else a ValueError is raised.
    """
    if has_duplicates(olds):
        raise ValueError("`olds` has duplicates")

    if has_duplicates(news):
        raise ValueError("`news` has duplicates")

    return {
        index_old: news.index(old)
        for index_old, old in [(olds.index(old), old) for old in olds if old in news]
    }


@ensure_python_version
def get_b_transitions(
    transitions: Dict[int, int], byte_source: bytes, byte_dest: bytes
) -> Dict[bytes, bytes]:
    return {
        byte_source + int2python_bytes(key): byte_dest + int2python_bytes(value)
        for key, value in transitions.items()
    }


@ensure_python_version
def are_functions_equivalent(l_func, r_func):
    """Return True if `l_func` and `r_func` are equivalent

    If Python version not in 3.{5, 6, 7}, a SystemError is raised.
    """
    l_code, r_code = l_func.__code__, r_func.__code__

    trans_co_consts = get_transitions(l_code.co_consts, r_code.co_consts)
    trans_co_names = get_transitions(l_code.co_names, r_code.co_names)
    trans_co_varnames = get_transitions(l_code.co_varnames, r_code.co_varnames)

    transitions = {
        **get_b_transitions(trans_co_consts, OpCode.LOAD_CONST, OpCode.LOAD_CONST),
        **get_b_transitions(trans_co_names, OpCode.LOAD_GLOBAL, OpCode.LOAD_GLOBAL),
        **get_b_transitions(trans_co_names, OpCode.LOAD_METHOD, OpCode.LOAD_METHOD),
        **get_b_transitions(trans_co_names, OpCode.LOAD_ATTR, OpCode.LOAD_ATTR),
        **get_b_transitions(trans_co_names, OpCode.STORE_ATTR, OpCode.STORE_ATTR),
        **get_b_transitions(trans_co_varnames, OpCode.LOAD_FAST, OpCode.LOAD_FAST),
        **get_b_transitions(trans_co_varnames, OpCode.STORE_FAST, OpCode.STORE_FAST),
    }

    l_instructions = get_instructions(l_func)
    r_instructions = get_instructions(r_func)

    new_l_instructions = tuple(
        transitions.get(instruction, instruction) for instruction in l_instructions
    )

    new_l_co_code = b"".join(new_l_instructions)

    co_code_cond = new_l_co_code == r_code.co_code
    co_consts_cond = set(l_code.co_consts) == set(r_code.co_consts)
    co_names_cond = set(l_code.co_names) == set(l_code.co_names)
    co_varnames_cond = set(l_code.co_varnames) == set(l_code.co_varnames)

    return co_code_cond and co_consts_cond and co_names_cond and co_varnames_cond


@ensure_python_version
def shift_instruction(instruction: bytes, qty: int) -> bytes:
    """Shift an instruction by qty.

    Examples:
    With Python 3.5:
    shift_instruction(b"d\x05\x00", 3) = b"d\x08\x00"

    With Python 3.{6, 7}:
    shift_instruction(b"d\x05", 3) = b"d\x08"

    If Python version not in 3.{5, 6, 7}, a SystemError is raised.
    """
    operation, *values = instruction
    return bytes((operation,)) + int2python_bytes(python_ints2int(values) + qty)


@ensure_python_version
def shift_instructions(instructions: Tuple[bytes], qty: int) -> Tuple[bytes]:
    """Shift JUMP_ABSOLUTE, JUMP_IF_FALSE_OR_POP,
    JUMP_IF_TRUE_OR_POP, POP_JUMP_IF_FALSE & POP_JUMP_IF_TRUE instructions by qty

    If Python version not in 3.{5, 6, 7}, a SystemError is raised.
    """
    return tuple(
        shift_instruction(instruction, qty)
        if bytes((instruction[0],))
        in (
            OpCode.JUMP_ABSOLUTE,
            OpCode.JUMP_IF_FALSE_OR_POP,
            OpCode.JUMP_IF_TRUE_OR_POP,
            OpCode.POP_JUMP_IF_FALSE,
            OpCode.POP_JUMP_IF_TRUE,
        )
        else instruction
        for instruction in instructions
    )


@ensure_python_version
def pin_arguments(func: FunctionType, arguments: dict):
    """Transform `func` in a function with no arguments.

    Example:

    def func(a, b):
        c = 4
        print(str(a) + str(c))

        return b

    The function returned by pin_arguments(func, {"a": 10, "b": 11}) is equivalent to:

    def pinned_func():
        c = 4
        print(str(10) + str(c))

        return 11

    This function is in some ways equivalent to functools.partials but with a faster
    runtime.

    `arguments` keys should be identical as `func` arguments names else a TypeError is
    raised.
    """

    if signature(func).parameters.keys() != set(arguments):
        raise TypeError("`arguments` and `func` arguments do not correspond")

    func_code = func.__code__
    func_co_consts = func_code.co_consts
    func_co_varnames = func_code.co_varnames

    new_co_consts = remove_duplicates(func_co_consts + tuple(arguments.values()))
    new_co_varnames = tuple(item for item in func_co_varnames if item not in arguments)

    trans_co_varnames2_co_consts = {
        func_co_varnames.index(key): new_co_consts.index(value)
        for key, value in arguments.items()
    }

    trans_co_varnames = get_transitions(func_co_varnames, new_co_varnames)

    transitions = {
        **get_b_transitions(
            trans_co_varnames2_co_consts, OpCode.LOAD_FAST, OpCode.LOAD_CONST
        ),
        **get_b_transitions(trans_co_varnames, OpCode.LOAD_FAST, OpCode.LOAD_FAST),
        **get_b_transitions(trans_co_varnames, OpCode.STORE_FAST, OpCode.STORE_FAST),
    }

    func_instructions = get_instructions(func)
    new_func_instructions = tuple(
        transitions.get(instruction, instruction) for instruction in func_instructions
    )

    new_co_code = b"".join(new_func_instructions)

    new_func = FunctionType(
        func.__code__,
        func.__globals__,
        func.__name__,
        func.__defaults__,
        func.__closure__,
    )

    nfcode = new_func.__code__

    new_func.__code__ = CodeType(
        0,
        0,
        len(new_co_varnames),
        nfcode.co_stacksize,
        nfcode.co_flags,
        new_co_code,
        new_co_consts,
        nfcode.co_names,
        new_co_varnames,
        nfcode.co_filename,
        nfcode.co_name,
        nfcode.co_firstlineno,
        nfcode.co_lnotab,
        nfcode.co_freevars,
        nfcode.co_cellvars,
    )

    return new_func


@ensure_python_version
def inline(pre_func: FunctionType, func: FunctionType, pre_func_arguments: dict):
    """Insert `prefunc` at the beginning of `func` and return the corresponding
    function.

    `pre_func` should not have a return statement (else a ValueError is raised).
    `pre_func_arguments` keys should be identical as `func` arguments names else a
    TypeError is raised.

    This approach takes less CPU instructions than the standard decorator approach.

    Example:

    def pre_func(b, c):
        a = "hello"
        print(a + " " + b + " " + c)

    def func(x, y):
        z = x + 2 * y
        return z ** 2

    The returned function corresponds to:

    def inlined(x, y):
        a = "hello"
        print(a)
        z = x + 2 * y
        return z ** 2
    """

    new_func = FunctionType(
        func.__code__,
        func.__globals__,
        func.__name__,
        func.__defaults__,
        func.__closure__,
    )

    if not has_no_return(pre_func):
        raise ValueError("`pre_func` returns something")

    pinned_pre_func = pin_arguments(pre_func, pre_func_arguments)
    pinned_pre_func_code = pinned_pre_func.__code__
    pinned_pre_func_co_consts = pinned_pre_func_code.co_consts
    pinned_pre_func_co_names = pinned_pre_func_code.co_names
    pinned_pre_func_co_varnames = pinned_pre_func_code.co_varnames
    pinned_pre_func_instructions = tuple(get_instructions(pinned_pre_func))
    pinned_pre_func_instructions_without_return = pinned_pre_func_instructions[:-2]

    func_code = func.__code__
    func_co_consts = func_code.co_consts
    func_co_names = func_code.co_names
    func_co_varnames = func_code.co_varnames

    func_instructions = tuple(get_instructions(func))
    shifted_func_instructions = shift_instructions(
        func_instructions, len(b"".join(pinned_pre_func_instructions_without_return))
    )

    new_co_consts = remove_duplicates(func_co_consts + pinned_pre_func_co_consts)
    new_co_names = remove_duplicates(func_co_names + pinned_pre_func_co_names)
    new_co_varnames = remove_duplicates(func_co_varnames + pinned_pre_func_co_varnames)

    trans_co_consts = get_transitions(pinned_pre_func_co_consts, new_co_consts)
    trans_co_names = get_transitions(pinned_pre_func_co_names, new_co_names)
    trans_co_varnames = get_transitions(pinned_pre_func_co_varnames, new_co_varnames)

    transitions = {
        **get_b_transitions(trans_co_consts, OpCode.LOAD_CONST, OpCode.LOAD_CONST),
        **get_b_transitions(trans_co_names, OpCode.LOAD_GLOBAL, OpCode.LOAD_GLOBAL),
        **get_b_transitions(trans_co_names, OpCode.LOAD_METHOD, OpCode.LOAD_METHOD),
        **get_b_transitions(trans_co_names, OpCode.LOAD_ATTR, OpCode.LOAD_ATTR),
        **get_b_transitions(trans_co_names, OpCode.STORE_ATTR, OpCode.STORE_ATTR),
        **get_b_transitions(trans_co_varnames, OpCode.LOAD_FAST, OpCode.LOAD_FAST),
        **get_b_transitions(trans_co_varnames, OpCode.STORE_FAST, OpCode.STORE_FAST),
    }

    new_pinned_pre_func_instructions = tuple(
        transitions.get(instruction, instruction)
        for instruction in pinned_pre_func_instructions_without_return
    )

    new_instructions = new_pinned_pre_func_instructions + shifted_func_instructions
    new_co_code = b"".join(new_instructions)

    nfcode = new_func.__code__

    new_func.__code__ = CodeType(
        nfcode.co_argcount,
        nfcode.co_kwonlyargcount,
        len(new_co_varnames),
        nfcode.co_stacksize,
        nfcode.co_flags,
        new_co_code,
        new_co_consts,
        new_co_names,
        new_co_varnames,
        nfcode.co_filename,
        nfcode.co_name,
        nfcode.co_firstlineno,
        nfcode.co_lnotab,
        nfcode.co_freevars,
        nfcode.co_cellvars,
    )

    return new_func
