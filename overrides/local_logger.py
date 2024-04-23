

"""
recbole.utils.logger
###############################
"""

import logging
import os
import colorlog
import re
import hashlib
from recbole.utils.utils import get_local_time, ensure_dir
from colorama import init

log_colors_config = {
    "DEBUG": "cyan",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red",
}


class RemoveColorFilter(logging.Filter):
    def filter(self, record):
        if record:
            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            record.msg = ansi_escape.sub("", str(record.msg))
        return True


def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Example:
        >>> logger = logging.getLogger(config)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    init(autoreset=True)

    # Ensure the base log directory exists
    base_log_dir = "./logs"
    if not os.path.exists(base_log_dir):
        os.mkdir(base_log_dir)

    # Create a dataset-specific log directory
    LOGROOT = os.path.join(base_log_dir, config['dataset'])
    if not os.path.exists(LOGROOT):
        os.mkdir(LOGROOT)

    logfilename = get_model_file_name(config) + '.log'
    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    if config["state"] is None or config["state"].lower() == "info":
        level = logging.INFO
    elif config["state"].lower() == "debug":
        level = logging.DEBUG
    elif config["state"].lower() == "error":
        level = logging.ERROR
    elif config["state"].lower() == "warning":
        level = logging.WARNING
    elif config["state"].lower() == "critical":
        level = logging.CRITICAL
    else:
        level = logging.INFO

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[sh, fh])

def get_model_file_name(config):
    logfilename = ''
    config_str = "".join([str(key) for key in config.final_config_dict.values()])
    md5 = hashlib.md5(config_str.encode(encoding="utf-8")).hexdigest()[:6]

    if len(config['train_type']) == 0:
        logfilename = "{}-{}-{}-{}".format(
            config["dataset"],'ordinary',get_local_time(), md5
        )
    else:
        info = ''
        for type in config['train_type']:
            info += type
            info += '-'
            info += str(config[type])
            info += '-'


        logfilename = "{}-{}{}-{}".format(
            config["dataset"], info, get_local_time(), md5
        )
    return logfilename

def get_model_file_name_without_dir(config):
    logfilename = ''
    config_str = "".join([str(key) for key in config.final_config_dict.values()])
    md5 = hashlib.md5(config_str.encode(encoding="utf-8")).hexdigest()[:6]

    if len(config['train_type']) == 0:
        logfilename = "{}-{}-{}-{}".format(
            config["dataset"],'ordinary',get_local_time(), md5
        )
    else:
        info = ''
        for type in config['train_type']:
            info += type
            info += '-'
            info += str(config[type])
            info += '-'


        logfilename = "{}-{}{}-{}".format(
            config["dataset"], info, get_local_time(), md5
        )
    return logfilename

if __name__ == '__main__':
    # if not os.path.exists("../log"):
    #     print("not exist")
    #     os.mkdir("../log")


    base_log_dir = "../log"
    if not os.path.exists(base_log_dir):
        print("not exist base_log_dir")
        os.mkdir(base_log_dir)