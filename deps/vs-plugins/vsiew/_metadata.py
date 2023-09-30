"""VapourSynth packages from Irrational Encoding Wizardry"""

__version__ = '2.3.8'

__author__ = 'Irrational Encoding Wizardry <wizards@encode.moe>'
__maintainer__ = 'Setsugen no ao <setsugen@setsugen.dev>'

__author_name__, __author_email__ = [x[:-1] for x in __author__.split('<')]
__maintainer_name__, __maintainer_email__ = [x[:-1] for x in __maintainer__.split('<')]

if __name__ == '__github__':
    print(__version__)
