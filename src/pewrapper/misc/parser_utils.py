import ctypes as ct
import os
import re
from configparser import ConfigParser, NoOptionError, NoSectionError, ParsingError
from enum import Enum
from typing import Any, Tuple

import pewrapper.misc.utils as utils
from navutils.logger import Logger
from pewrapper.types.gps_time_wrapper import GPS_Time
from rlnav.types.reference_types import ReferenceMode, ReferenceType


class ConfigDefaultValue(Enum):
    default_value_false = 0
    default_value_true = 1
    default_value_none = 2


def parse_session_file(
    filename: str,
    verbose=True,
) -> Tuple[bool, str, str, str, str, GPS_Time, GPS_Time]:
    result = True
    config_PE_path = ""
    wrapper_file_path = ""
    config_tracing_path = ""
    addInfo = ""
    initial_epoch_session = GPS_Time()
    final_epoch_session = GPS_Time()

    if verbose:
        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.MAIN,
            f" Parsing session file: {filename}",
        )

    if not os.path.isfile(filename):
        addInfo = "Session file doesn't exist."
        return (
            False,
            config_PE_path,
            wrapper_file_path,
            config_tracing_path,
            addInfo,
            initial_epoch_session,
            final_epoch_session,
        )

    pt = ConfigParser()
    pt.read(filename)

    result, addInfo, initial_epoch_session = ParseSessionEpoch(
        pt, "InitialEpoch", "Initial Epoch", addInfo, verbose=verbose
    )
    if not result:
        return (
            False,
            config_PE_path,
            wrapper_file_path,
            config_tracing_path,
            addInfo,
            initial_epoch_session,
            final_epoch_session,
        )

    success, addInfo, final_epoch_session = ParseSessionEpoch(
        pt, "FinalEpoch", "Final Epoch", addInfo, verbose=verbose
    )
    result = result and success
    if not result:
        return (
            False,
            config_PE_path,
            wrapper_file_path,
            config_tracing_path,
            addInfo,
            initial_epoch_session,
            final_epoch_session,
        )

    success, addInfo, config_PE_path = ParseSessionField(
        config_PE_path,
        pt,
        "FilesPaths.config_PE_path",
        "Config Wrapper file path",
        addInfo,
        verbose=verbose,
    )
    result = result and success
    if not result:
        return (
            False,
            config_PE_path,
            wrapper_file_path,
            config_tracing_path,
            addInfo,
            initial_epoch_session,
            final_epoch_session,
        )

    success, addInfo, wrapper_file_path = ParseSessionField(
        wrapper_file_path,
        pt,
        "FilesPaths.wrapper_conversor_data_path",
        "Wrapper Conversor Data file path",
        addInfo,
        verbose=verbose,
    )
    result = result and success
    if not result:
        return (
            False,
            config_PE_path,
            wrapper_file_path,
            config_tracing_path,
            addInfo,
            initial_epoch_session,
            final_epoch_session,
        )

    success, addInfo, config_tracing_path = ParseSessionField(
        config_tracing_path,
        pt,
        "FilesPaths.config_tracing_path",
        "Config tracing file path",
        addInfo,
        verbose=verbose,
    )
    result = result and success
    if not result:
        return (
            False,
            config_PE_path,
            wrapper_file_path,
            config_tracing_path,
            addInfo,
            initial_epoch_session,
            final_epoch_session,
        )

    return (
        True,
        config_PE_path,
        wrapper_file_path,
        config_tracing_path,
        addInfo,
        initial_epoch_session,
        final_epoch_session,
    )


def write_session_file(
    filename_src: str, filename_dst: str, change_session_epochs=None, verbose=True
) -> bool:
    if not os.path.isfile(filename_src):
        if verbose:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.WRITER,
                f"[write_session_file] File not found: {filename_src}",
            )
        return False

    pt = ConfigParser()
    pt.read(filename_src)

    if change_session_epochs is not None and change_session_epochs:
        for section in ["InitialEpoch", "FinalEpoch"]:
            epoch = change_session_epochs[section].calendar_column_str_d()
            year, month, day, hour, minute, second = re.sub(" +", " ", epoch).split(" ")
            pt.set(section, "year", f"{int(year):04d}")
            pt.set(section, "month", f"{int(month):02d}")
            pt.set(section, "day", f"{int(day):02d}")
            pt.set(section, "hour", f"{int(hour):02d}")
            pt.set(section, "minute", f"{int(minute):02d}")
            pt.set(section, "second", f"{int(second.split('.')[0]):02d}")

    pt.write(open(filename_dst, "w"), space_around_delimiters=False)

    if verbose:
        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.MAIN,
            f"Session file written: {filename_dst}",
        )

    return True


def parse_subsessions_file(filename: str, subscenario: str, verbose=True) -> dict:
    subsessions_dict = {}

    if not os.path.isfile(filename):
        if verbose:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                f"[parse_subsessions_file] File not found: {filename}",
            )
        return subsessions_dict

    pt = ConfigParser()
    pt.read(filename)

    if verbose:
        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.READER,
            f"Loaded subsection: {subscenario}",
        )

    if subscenario in pt.sections():
        success_initial, initial_epoch = _parse_subsession_epoch(
            pt, subscenario, prefix="initial", verbose=verbose
        )
        success_final, final_epoch = _parse_subsession_epoch(
            pt, subscenario, prefix="final", verbose=verbose
        )

        if success_initial and success_final:
            subsessions_dict = {
                "InitialEpoch": initial_epoch,
                "FinalEpoch": final_epoch,
            }
        else:
            if verbose:
                Logger.log_message(
                    Logger.Category.WARNING,
                    Logger.Module.MAIN,
                    f"Incomplete data in section [{subscenario}], is omitted.",
                )

    return subsessions_dict


def ParseSessionEpoch(
    pt: ConfigParser,
    field: str,
    label: str,
    addInfo: str,
    verbose=True,
) -> Tuple[bool, str, GPS_Time]:
    epoch_session = GPS_Time()

    try:
        year = pt.getint(section=field, option="year")
        month = pt.getint(section=field, option="month")
        day = pt.getint(section=field, option="day")
        hour = pt.getint(section=field, option="hour")
        minute = pt.getint(section=field, option="minute")
        second = pt.getint(section=field, option="second")

        epoch_session = GPS_Time(
            year=year, month=month, day=day, hour=hour, minute=minute, second=second
        )

    except (NoSectionError, NoOptionError):
        addInfo = f"Session field for {label} not written in session file"
        return False, addInfo, epoch_session
    except ParsingError:
        addInfo = f"Session field for {label} bad formed in session file"
        return False, addInfo, epoch_session

    if verbose:
        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.READER,
            f"{label}: {epoch_session.calendar_column_str_d()}",
        )

    return True, addInfo, epoch_session


def _parse_subsession_epoch(
    pt: ConfigParser, section: str, prefix: str = "initial", verbose=True
) -> tuple:
    epoch_session = GPS_Time()
    fields = ["year", "month", "day", "hour", "minute", "second"]

    try:
        kwargs = {}
        for f in fields:
            key = f"{f}_{prefix}"
            kwargs[f] = pt.getint(section, key)

        epoch_session = GPS_Time(
            year=kwargs["year"],
            month=kwargs["month"],
            day=kwargs["day"],
            hour=kwargs["hour"],
            minute=kwargs["minute"],
            second=kwargs["second"],
        )

        if verbose:
            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.READER,
                f"{section} - {prefix}_epoch: {epoch_session.calendar_column_str_d()}",
            )

        return True, epoch_session

    except (NoSectionError, NoOptionError):
        if verbose:
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.READER,
                f"Missing fields for '{prefix}' epoch in section [{section}]",
            )
        return False, epoch_session
    except ParsingError:
        if verbose:
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.READER,
                f"Bad format for '{prefix}' epoch in section [{section}]",
            )
        return False, epoch_session


def ParseSessionField(
    value, pt: ConfigParser, field: str, label: str, addInfo: str, verbose=True
) -> Tuple[bool, str, str]:
    try:
        section, option = field.split(".", 1)
        value = pt.get(section, option)

    except (NoSectionError, NoOptionError):
        addInfo = f"Session field for {label} not written in session file"
        return False, addInfo, value
    except ParsingError:
        addInfo = f"Session field for {label} bad formed in session file"
        return False, addInfo, value

    if verbose:
        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.READER,
            f" Session: {label} set to {value}",
        )

    return True, addInfo, value


def ParseConfiguration(
    value,
    pt: ConfigParser,
    field: str,
    label: str,
    addInfo: str,
    check_invalid=False,
    invalid_value=None,
    is_default=False,
    default_value=None,
    verbose=True,
) -> Tuple[bool, str, str]:
    try:
        section, option = field.split(".", 1)
        value = utils.get_id(type(invalid_value)(value), pt.get(section, option))
    except (NoSectionError, NoOptionError):
        if is_default:
            value = ct.c_uint32(default_value)
        else:
            addInfo = f"Configuration-node for {label} not written in config file"
            return False, addInfo, value
    except ParsingError:
        addInfo = f"Configuration data for {label} bad formed in config file"
        return False, addInfo, value

    if check_invalid:
        if value == ct.c_uint32(invalid_value):
            if verbose:
                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.READER,
                    f" Configuration: Invalid value for {label} {pt.get(section=field.split('.')[0], option=field.split('.')[-1])}",
                )
            addInfo = f" Configuration: Invalid value for {label}"
            return False, addInfo, value

    if verbose:
        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.READER,
            f" Configuration: {label}  set to {utils.get_id(type(invalid_value)(value.value))}",
        )

    return True, addInfo, value


def ParseConfigurationField(
    value,
    pt: ConfigParser,
    field: str,
    label: str,
    addInfo: str,
    is_default=False,
    default_value=None,
    verbose=True,
) -> Tuple[bool, str, Any]:
    if isinstance(value, (ct.c_double * 3)):
        try:
            section, option = field.split(".", 1)
            value[0] = ct.c_double(pt.getfloat(section, option + "_x"))
            value[1] = ct.c_double(pt.getfloat(section, option + "_y"))
            value[2] = ct.c_double(pt.getfloat(section, option + "_z"))
        except (NoSectionError, NoOptionError):
            addInfo = f"Configuration-node for {label} not written in config file"
            return False, addInfo, value
        except ParsingError:
            addInfo = f"Configuration data for {label} bad formed in config file"
            return False, addInfo, value

        if verbose:
            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.READER,
                f" Configuration: {label}  set to X = {value[0]}, Y = {value[1]}, Z = {value[2]}",
            )

        return True, addInfo, value
    else:
        try:
            section, option = field.split(".", 1)
            value = type(value)(pt.get(section, option))
        except (NoSectionError, NoOptionError):
            if is_default:
                value = default_value
            else:
                addInfo = f"Configuration-node for {label} not written in config file"
                return False, addInfo, value
        except ParsingError:
            addInfo = f"Configuration data for {label} bad formed in config file"
            return False, addInfo, value

        if verbose:
            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.READER,
                f" Configuration: {label}  set to {value}",
            )

        return True, addInfo, value


def ParseBoolConfigurationField(
    value: bool,
    pt: ConfigParser,
    field: str,
    label: str,
    addInfo: str,
    default_value: ConfigDefaultValue = ConfigDefaultValue.default_value_none,
    verbose=True,
) -> Tuple[bool, str, bool]:
    try:
        section, option = field.split(".", 1)
        value = pt.getboolean(section, option)
    except (NoSectionError, NoOptionError):
        if default_value == ConfigDefaultValue.default_value_true:
            value = True
        elif default_value == ConfigDefaultValue.default_value_false:
            value = False
        else:
            addInfo = f"Configuration-node for {label} not written in config file"
            return False, addInfo, value
    except ParsingError:
        addInfo = f"Configuration data for {label} bad formed in config file"
        return False, addInfo, value

    if verbose:
        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.READER,
            f" Configuration: {label}  set to {f'{value}'.upper()}",
        )

    return True, addInfo, value


def parse_scenario_file(
    filename: str,
):
    result = True
    reference_mode = None
    reference = None

    try:
        with open(filename, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue

                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    if key == "REFERENCE_MODE":  # [STATIC, KINEMATIC, NONE]
                        if value == "KINEMATIC":
                            reference_mode = ReferenceMode.KINEMATIC
                        elif value == "STATIC":
                            reference_mode = ReferenceMode.STATIC
                        else:
                            Logger.log_message(
                                Logger.Category.WARNING,
                                Logger.Module.READER,
                                f"No reference mode configured, using default KINEMATIC",
                            )
                            reference_mode = ReferenceMode.KINEMATIC

                    elif key == "REFERENCE":  # [span_file, rtk_file, positions_file]
                        if value == "span_file":
                            reference = ReferenceType.Kinematic.SPAN_FILE
                        elif value == "rtk_file":
                            reference = ReferenceType.Kinematic.RTK_FILE
                        elif value == "positions_file":
                            reference = ReferenceType.Kinematic.RTPPP_GSHARP
                        else:
                            Logger.log_message(
                                Logger.Category.WARNING,
                                Logger.Module.READER,
                                f"No reference type configured, using default SPAN_FILE",
                            )
                            reference = ReferenceType.Kinematic.SPAN_FILE

    except:
        result = False

    return result, reference_mode, reference
