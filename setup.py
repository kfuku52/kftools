from setuptools import setup, find_packages

setup(
        name             = 'kftools',
        version          = "0.1",
        description      = 'Utility tools for personal use',
        license          = "BSD 3-clause License",
        author           = "Kenji Fukushima",
        author_email     = 'kfuku52@gmail.com',
        url              = 'https://github.com/kfuku52/kftools.git',
        keywords         = '',
        packages         = find_packages(),
        install_requires = ['ete3','numpy','pandas','matplotlib','seaborn','scipy'],
        )