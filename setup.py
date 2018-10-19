#- 
# setup.py
#-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'PITAS: (P)owerspectrum (I)n (T)racts (A)lgorithm on the (S)phere',
    'author':  "Thibaut Louis, Dongwon 'DW' HAN",
    'url': 'https://github.com/dwhan89/cusps',
    'download_url': 'https://github.com/dwhan89/pitas',
    'author_email': 'dongwon.han@stonybrook.edu',
    'version': '1.1.1',
    'install_requires': [
        'numpy',
        'matplotlib',
        'pixell'
        ],
    'packages': [
        'pitas',
        'pitas.wigner',
        'pitas.mcm_core'
        ],
    'scripts': [],
    'name': 'pitas'
}

setup(**config)

