from typing import Tuple, Union, Optional, TypeVar
import numpy.typing as npt
import numpy as np

U = TypeVar("U")


def generate_table_e2e(
    table: npt.NDArray[np.uint32],
) -> Tuple[bool, npt.NDArray[np.uint32]]:
    pol_crc_e2e = 0xC8DF352F

    for i in range(256):
        aux = i
        for j in range(8):
            if aux & 1 > 0:
                aux = pol_crc_e2e ^ (aux >> 1)
            else:
                aux >>= 1
        table[i] = aux

    return True, table


def compute_crc_e2e(msg: bytes, length: int, table: npt.NDArray[np.uint32]) -> int:
    value = 0 ^ 0xFFFFFFFF

    for i in range(length):
        value = table[(value ^ msg[i]) & 0xFF] ^ (value >> 8)

    return value ^ 0xFFFFFFFF


def _check_counter(
    current_counter: U,
    internal_counter: U,
    max_counter_value: U,
    rollover_threshold: U,
    valid_counter: bool,
    counter_rollover: bool,
) -> Tuple[bool, bool]:
    counter_rollover = False

    if current_counter > internal_counter:
        valid_counter = True

    elif current_counter < internal_counter:
        updated_counter = (max_counter_value - internal_counter) + current_counter
        valid_counter = updated_counter <= rollover_threshold
        counter_rollover = valid_counter

    else:
        pass

    return valid_counter, counter_rollover


def check_counter(
    current_counter: U,
    internal_counter: U,
    max_counter_value: U,
    rollover_threshold: U,
    valid_counter: bool,
    counter_rollover: Optional[bool] = None,
    empty_meas: Optional[bool] = None,
) -> Union[Tuple[bool, bool], Tuple[bool, bool, U]]:
    if empty_meas is None:
        return _check_counter(
            current_counter,
            internal_counter,
            max_counter_value,
            rollover_threshold,
            valid_counter,
            counter_rollover if counter_rollover is not None else False,
        )

    else:
        assert counter_rollover is not None

        if not empty_meas:
            valid_counter, counter_rollover = _check_counter(
                current_counter,
                internal_counter,
                max_counter_value,
                rollover_threshold,
                valid_counter,
                counter_rollover,
            )
            internal_counter = current_counter

        return valid_counter, counter_rollover, internal_counter
