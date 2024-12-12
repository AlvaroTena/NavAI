import builtins
import contextlib
import io
import logging
import sys
from typing import Text, Union

import tqdm
from neptune import Run
from neptune.integrations.python_logger import NeptuneHandler

global _print
_print = print


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)


class Logger:
    def __init__(
        self,
        level: Union[int, str, any],
        npt_run: Run = None,
        stdout: bool = True,
        filename: Text = None,
        handle_prints: bool = False,
    ):
        handlers = []

        (
            handlers.append(
                logging.FileHandler(
                    filename=filename,
                    mode="w",
                    encoding="utf-8",
                )
            )
            if filename
            else None
        )
        handlers.append(logging.StreamHandler(stream=sys.stdout)) if stdout else None
        handlers.append(NeptuneHandler(run=npt_run)) if npt_run is not None else None

        logging.basicConfig(
            level=level,
            format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            handlers=handlers,
        )

        logging.getLogger("matplotlib.font_manager").disabled = True

        if handle_prints:
            builtins.print = logging.debug

    def __del__(self):
        Logger.resetPrint()

    @staticmethod
    def getLogger():
        return logging.getLogger("mlnav")

    @staticmethod
    def getTqdmLogger():
        return TqdmToLogger(logger=logging.getLogger("mlnav"), level=logging.DEBUG)

    @staticmethod
    def resetPrint():
        builtins.print = _print


@contextlib.contextmanager
def tqdm_progress(initial, total):
    with tqdm.tqdm(
        total=total,
        file=Logger.getTqdmLogger(),
        postfix={"best loss": "?"},
        disable=False,
        dynamic_ncols=True,
        unit="trial",
        initial=initial,
    ) as pbar:
        yield pbar
