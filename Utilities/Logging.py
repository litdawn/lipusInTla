import logging


class LOG:
    level = ""

    def __init__(self, level="info"):
        logging.basicConfig(level=logging.INFO)
        pass

    def debug(self, msg):
        logging.debug(msg)

    def info(self, msg):
        logging.info(msg)

    def error(self, msg):
        logging.error(msg)


log = LOG()
