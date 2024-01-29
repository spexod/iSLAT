from setuptools import setup

setup(
    name='iSLAT',
    version='3.10.00',
    author='M.Johnson, E.Jellison, A.Banzatti, S.Bruderer',
    author_email='banzatti@txstate.edu',
    description='Interactive tool for the visualization, exploration, and analysis of molecular spectra.',
    packages=['iSLAT'],
    url="https://github.com/spexod/iSLAT",
    install_requires=[
        'numpy',
        'astropy',
        'tk',
        'lmfit',
        'pandas',
        'scipy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    package_directory={'iSLAT': 'iSLAT'}
)
