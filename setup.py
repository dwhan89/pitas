#- 
# setup.py
#-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'CUSPS: Curved Sky Power Spectrum',
    'author': "Dongwon 'DW' HAN",
    'url': 'https://github.com/dwhan89/cusps',
    'download_url': 'https://github.com/dwhan89/cusps',
    'author_email': 'dongwon.han@stonybrook.edu',
    'version': '0.1',
    'install_requires': [
        'numpy',
        'matplotlib',
        'enlib',
        'sympy >= 1.0'
        ],
    'packages': [
        'cusps',
        'cusps.wigner'
        ],
    'scripts': [],
    'name': 'cusps'
}

setup(**config)

