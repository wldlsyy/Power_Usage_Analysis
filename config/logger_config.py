import logging

def setup_logger(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s : %(levelname)s : [%(filename)s:%(lineno)d - %(funcName)s()] : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )