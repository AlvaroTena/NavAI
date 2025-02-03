import ctypes as ct
import os
from typing import ByteString

from navutils.logger import Logger
from pewrapper.api.common_api_types import (
    SIZE_BINEPHEM,
    Log_Handle,
    LogCategoryPE,
    f_handle_bin_ephem_NVM,
)

MAX_LOG_SIZE = 102400


@f_handle_bin_ephem_NVM
def parse_binEphem(ephemData: ct.c_void_p, amountToRead: ct.c_size_t):
    result = True
    fullPathFilename = "data/scenarios/Files_Test/binEphem.405"

    try:
        with open(
            fullPathFilename,
            "rb",
        ) as ephemerisFile:
            data = ephemerisFile.read(amountToRead)
            if len(data) < amountToRead:
                result = False
            else:
                ct.memmove(ephemData, data, len(data))
    except IOError:
        result = False

    return result


@Log_Handle
def PE_LogWrapper(
    category: ct.c_uint32,
    eventParticulars: ct.c_char_p,
    fileName: ct.c_char_p,
    function: ct.c_char_p,
    codeLine: ct.c_uint16,
    use_AI: ct.c_bool,
):
    level = {
        LogCategoryPE.NONE.value: Logger.Category.NOTSET,
        LogCategoryPE.TRACE.value: Logger.Category.TRACE,
        LogCategoryPE.DEBUG.value: Logger.Category.DEBUG,
        LogCategoryPE.INFO.value: Logger.Category.INFO,
        LogCategoryPE.WARNING.value: Logger.Category.WARNING,
        LogCategoryPE.error.value: Logger.Category.ERROR,
    }.get(category, Logger.Category.ERROR)
    message = eventParticulars.decode("utf-8")
    # Logger.log_message(level, Logger.Module.PE, f"{message}", use_AI=use_AI)
