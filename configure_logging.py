"""
Intended to be put into ~/.ipython/profile_default/startup/ when using Jupyter.
If run outside of Jupyter, needs to be imported manually to configure the logging.
"""
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s : %(levelname)s : %(message)s", "%Y-%m-%d %H:%M:%S")
)
logger.addHandler(handler)
