import ast
import re
from pathlib import Path

from setuptools import setup, find_packages

with open(Path('kftools') / '__init__.py', encoding='utf-8') as f:
    match = re.search(r'__version__\s+=\s+(.*)', f.read())
version = str(ast.literal_eval(match.group(1)))

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='kftools',
    version=version,
    description='Utility tools for personal use',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="MIT",
    author="Kenji Fukushima",
    author_email='kfuku52@gmail.com',
    url='https://github.com/kfuku52/kftools.git',
    keywords='',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'ete4>=4.1.1',
        'numpy>=1.23',
        'pandas>=1.5',
        'matplotlib>=3.6',
        'scipy>=1.9',
        'statsmodels>=0.13.5',
    ],
    extras_require={
        'dev': [
            'build>=1.2',
            'coverage[toml]>=7',
            'mypy>=1.10',
            'pip-audit>=2.7',
            'pytest>=8',
            'ruff>=0.8',
            'twine>=5',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
