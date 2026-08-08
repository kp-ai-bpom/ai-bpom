import logging
import sys


def setup_logger(logger_name: str = "bpom_ai_service") -> logging.Logger:
    """
    Setup dasar untuk custom logger aplikasi.
    """
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Opsional: Jika Anda ingin menyimpan log ke file (misal: app.log)
        # file_handler = logging.FileHandler("app.log")
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)

    return logger


log = setup_logger()
