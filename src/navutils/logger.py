import logging
import os
import queue
import sys
from enum import Enum, IntEnum
from logging.handlers import QueueHandler, QueueListener
from threading import Lock

from navutils.singleton import Singleton
from neptune import Run
from neptune.integrations.python_logger import NeptuneHandler


def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        return
        # raise AttributeError("{} already defined in logging module".format(levelName))
    if hasattr(logging, methodName):
        return
        # raise AttributeError("{} already defined in logging module".format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        return
        # raise AttributeError("{} already defined in logger class".format(methodName))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


class WrapperLogFilter(logging.Filter):
    def __init__(self, use_ai):
        super().__init__()
        self.use_ai = use_ai

    def filter(self, record):
        if getattr(record, "use_ai", None) is None:
            return True
        return getattr(record, "use_ai", False) == self.use_ai


class Logger(metaclass=Singleton):
    _instance = None

    class Category(IntEnum):
        CRITICAL = logging.CRITICAL
        ERROR = logging.ERROR
        WARNING = logging.WARNING
        INFO = logging.INFO
        DEBUG = logging.DEBUG
        TRACE = 5
        NOTSET = logging.NOTSET

        def get_log_category(cat_str: str):
            if cat_str == "DEBUG":
                cat = Logger.Category.DEBUG
            elif cat_str == "INFO":
                cat = Logger.Category.INFO
            elif cat_str == "WARNING":
                cat = Logger.Category.WARNING
            elif cat_str in ["ERROR", "JENKINS"]:
                cat = Logger.Category.ERROR
            else:
                cat = Logger.Category.NOTSET
                return False, cat
            return True, cat

    class Module(Enum):
        NONE = 0
        RECEIVER = 1
        ALGORITHM = 2
        CRYPTO = 3
        CONFIG = 4
        INTERNET = 5
        IRIDIUM = 6
        COMMUNICATION = 7
        MMI = 8
        DB = 9
        READER = 10
        WRITER = 11
        MAIN = 12
        IMU = 13
        STATE_MACHINE = 14
        PE = 15
        LOGGER = 16
        DATAPROCESSOR = 17
        ENV = 18
        WRAPPER = 19
        REWARD = 20
        MONITOR = 21

        def get_module_string(mod):
            module_dict = {
                Logger.Module.NONE: "No specific mod.",
                Logger.Module.RECEIVER: "Receiver handler",
                Logger.Module.ALGORITHM: "Algorithm manager",
                Logger.Module.CRYPTO: "Crypto. module",
                Logger.Module.CONFIG: "Config. functions",
                Logger.Module.INTERNET: "Internet manager",
                Logger.Module.IRIDIUM: "Iridium manager",
                Logger.Module.COMMUNICATION: "Comm. manager",
                Logger.Module.MMI: "MMI handler",
                Logger.Module.DB: "Database",
                Logger.Module.READER: "Reader",
                Logger.Module.WRITER: "Writer",
                Logger.Module.MAIN: "Main functions",
                Logger.Module.IMU: "IMU manager",
                Logger.Module.STATE_MACHINE: "State machine",
                Logger.Module.PE: "Position Engine",
                Logger.Module.LOGGER: "Logger",
                Logger.Module.DATAPROCESSOR: "Data Processor",
                Logger.Module.ENV: "RL Environment",
                Logger.Module.WRAPPER: "Wrapper",
                Logger.Module.REWARD: "Reward manager",
                Logger.Module.MONITOR: "Monitoring",
            }
            mod_str = module_dict.get(mod, "???").ljust(17)

            return mod_str

    def __init__(
        self,
        output_path: str,
        log_queue=None,
    ):
        """Initializes the Logger instance."""
        if Logger._instance is not None:
            return
        else:
            Logger._instance = self

            self.log_dir = os.path.join(output_path, "pe_log_files")
            os.makedirs(self.log_dir, exist_ok=True)

            self.handlers = []
            if log_queue is None:
                from multiprocessing import Manager

                manager = Manager()
                self.log_queue = manager.Queue()
                self.is_parent = True
            else:
                self.log_queue = log_queue
                self.is_parent = False

            self.queue_handler = QueueHandler(self.log_queue)
            self._initialize_handlers()

            self.log_lock = Lock()
            logging.basicConfig(
                level=logging.ERROR,
                format="%(asctime)s.%(msecs)03d [%(levelname)-8s] %(message)s",
                datefmt="%Y/%m/%d %H:%M:%S",
                handlers=[self.queue_handler],
            )
            self.logger = logging.getLogger(__name__)

            if self.is_parent:
                self.queue_listener = QueueListener(self.log_queue, *self.handlers)
                self.queue_listener.start()
            else:
                self.queue_listener = None

            addLoggingLevel("TRACE", Logger.Category.TRACE)

    def _initialize_handlers(self):
        """Initializes log handlers."""
        regular_handler = self._create_file_handler("wrapper_session.log", False)
        self.handlers.append(regular_handler)

        self.stdout_handler = logging.StreamHandler(stream=sys.stdout)
        self.handlers.append(self.stdout_handler)

    def _create_file_handler(self, filename, use_ai):
        """Creates a file handler with the specified filename and filter."""
        file_handler = logging.FileHandler(
            filename=os.path.join(self.log_dir, filename), mode="w", encoding="utf-8"
        )
        # file_handler.addFilter(WrapperLogFilter(use_ai=use_ai))
        return file_handler

    @staticmethod
    def reconfigure_child(log_queue):
        instance = Logger.get_instance()
        with instance.log_lock:
            instance.log_queue = log_queue
            instance.is_parent = False
            instance.queue_listener = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Logger("./")
        return cls._instance

    @staticmethod
    def log_message(level: Category, module: Module, message: str, *args):
        """Logs a message at the specified logging level."""
        instance = Logger.get_instance()
        with instance.log_lock:
            if level >= instance.logger.getEffectiveLevel():
                log_record = logging.LogRecord(
                    name=instance.logger.name,
                    level=level,
                    pathname="",
                    lineno=0,
                    msg=f"[{Logger.Module.get_module_string(module)}] {message.strip()}",
                    args=args,
                    exc_info=None,
                )
                instance.logger.handle(log_record)

    @staticmethod
    def get_category():
        """Gets the current logging category level."""
        instance = Logger.get_instance()
        return Logger.Category(instance.logger.getEffectiveLevel())

    @staticmethod
    def get_queue():
        """Gets the current logging queue."""
        instance = Logger.get_instance()
        return instance.log_queue

    @staticmethod
    def set_category(cat: str):
        """Sets the logging category level."""
        _, category = Logger.Category.get_log_category(cat)
        instance = Logger.get_instance()
        with instance.log_lock:
            instance.logger.setLevel(category)

    @staticmethod
    def reset():
        """Resets the logging level to ERROR."""
        instance = Logger.get_instance()
        with instance.log_lock:
            instance.logger.setLevel(Logger.Category.ERROR)

    @staticmethod
    def stop_listener():
        """Stops the QueueListener gracefully."""
        instance = Logger.get_instance()
        if hasattr(instance, "queue_listener"):
            instance.queue_listener.stop()
            instance.queue_listener.join()

    def __del__(self):
        """Destructor to ensure the QueueListener is stopped."""
        self.stop_listener()
