import logging
import sys

from heart_classification_model.config.core import PACKAGE_ROOT, config

sys.path.append(
    "/Users/khvandima/Documents/Programming/KS_AI_JLR_18/ml_practice/Heart_Disease_classification/"
)

logging.getLogger(config.app_config.package_name).addHandler(logging.NullHandler())

with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()
