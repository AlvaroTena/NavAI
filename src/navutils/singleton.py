from threading import Lock

_instance_lock = Lock()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        global _instance_lock
        if cls not in cls._instances:
            with _instance_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(Singleton, cls).__call__(
                        *args, **kwargs
                    )
        return cls._instances[cls]
