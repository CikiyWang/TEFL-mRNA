import signal
from pkg_resources import get_distribution


signal.signal(signal.SIGINT, lambda x, y: exit(0))

name = 'TEFL-mRNA'
__version__ = get_distribution(name).version