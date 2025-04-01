from dataclasses import dataclass


@dataclass
class E2EMsgValidity:
    headerDataId = True
    crc = True
    aliveCounter = True


def get_e2e_msg_validity(e2emsgvalidity: E2EMsgValidity):
    return (
        e2emsgvalidity.headerDataId
        and e2emsgvalidity.crc
        and e2emsgvalidity.aliveCounter
    )
