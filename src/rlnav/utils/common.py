import re

import pewrapper.types.constants as pe_const

MAX_SATS_PER_CONSTEL = {
    pe_const.E_CONSTELLATION.E_CONSTEL_GPS: pe_const.MAX_SATS_GPS,
    pe_const.E_CONSTELLATION.E_CONSTEL_GAL: pe_const.MAX_SATS_GAL,
    pe_const.E_CONSTELLATION.E_CONSTEL_BDS: pe_const.MAX_SATS_BDS,
}

_SUBSCENARIO_RE = re.compile(
    r"_(?P<idx>\d+)_(?P<h>(?:LOWH|MIDH|HIGHH))_(?P<v>(?:LOWV|MIDV|HIGHV))$"
)


def get_global_sat_idx(cons_idx: int, sat_idx: int, freq_idx: int = None) -> int:
    global_offset = sum(
        MAX_SATS_PER_CONSTEL.get(constel, 0)
        for constel in list(MAX_SATS_PER_CONSTEL.keys())[:cons_idx]
    )
    if freq_idx is not None:
        global_offset += freq_idx * pe_const.MAX_SATS

    return global_offset + sat_idx


def is_scenario_subset(scenario_name: str) -> bool:
    return bool(_SUBSCENARIO_RE.search(scenario_name))


def get_parent_scenario_name(scenario_name: str) -> str:
    if is_scenario_subset(scenario_name):
        return _SUBSCENARIO_RE.sub("", scenario_name)
    return scenario_name
