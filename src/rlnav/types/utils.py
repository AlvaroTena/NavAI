import re

import pewrapper.types.constants as pe_const

MAX_SATS_PER_CONSTEL = {
    pe_const.E_CONSTELLATION.E_CONSTEL_GPS: pe_const.MAX_SATS_GPS,
    pe_const.E_CONSTELLATION.E_CONSTEL_GAL: pe_const.MAX_SATS_GAL,
    pe_const.E_CONSTELLATION.E_CONSTEL_BDS: pe_const.MAX_SATS_BDS,
}


def get_global_sat_idx(cons_idx: int, sat_idx: int, freq_idx: int = None) -> int:
    global_offset = sum(
        MAX_SATS_PER_CONSTEL.get(constel, 0)
        for constel in list(MAX_SATS_PER_CONSTEL.keys())[:cons_idx]
    )
    if freq_idx is not None:
        global_offset += freq_idx * pe_const.MAX_SATS

    return global_offset + sat_idx


def get_parent_scenario_name(scenario_name):
    pattern = r"^(.*?\d{8})"
    pattern_match = re.search(pattern, scenario_name)

    if pattern_match:
        return pattern_match.group(1)
    else:
        return scenario_name


def is_scenario_subset(sceario_name):
    if sceario_name:
        parent = get_parent_scenario_name(sceario_name)
        return len(sceario_name) > len(parent)

    else:
        return False
