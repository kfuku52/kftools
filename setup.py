import ast
import re
from pathlib import Path

from setuptools import setup, find_packages

with open(Path('kftools') / '__init__.py') as f:
    match = re.search(r'__version__\s+=\s+(.*)', f.read())
version = str(ast.literal_eval(match.group(1)))

setup(
    name='kftools',
    version=version,
    description='Utility tools for personal use',
    license="MIT License",
    author="Kenji Fukushima",
    author_email='kfuku52@gmail.com',
    url='https://github.com/kfuku52/kftools.git',
    keywords='',
    packages=find_packages(),
    install_requires=['ete3', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'statsmodels'],
)
