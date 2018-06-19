#- 
# setup.py
#-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'CUSPS: Curved Sky Power Spectrum',
    'author':  "Thibaut Louis, Dongwon 'DW' HAN",
    'url': 'https://github.com/dwhan89/cusps',
    'download_url': 'https://github.com/dwhan89/cusps',
    'author_email': 'dongwon.han@stonybrook.edu',
    'version': '0.1',
    'install_requires': [
        'numpy',
        'matplotlib',
        'enlib',
        ],
    'packages': [
        'cusps',
        'cusps.wigner',
        'cusps.mcm_core'
        ],
    'scripts': [],
    'name': 'cusps'
}

setup(**config)

