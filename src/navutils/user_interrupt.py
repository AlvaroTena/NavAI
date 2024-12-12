class UserInterruptException(Exception):
    def __init__(self, msg: str):
        self.msg_ = msg

    def get_msg(self):
        return self.msg_


def signal_handler(signal: int):
    raise UserInterruptException("Execution terminated by user")
