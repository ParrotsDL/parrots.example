import logging
import parrots
from parrots.log_utils import basicConfig


assert len(logging.root.handlers) == 1
assert logging.root.handlers[0].level == 20
basicConfig()
assert len(logging.root.handlers) == 1
assert logging.root.handlers[0].level == 0
print("test successfully!")