from setuptools import setup, find_packages

exec(open('quantum_electron/_version.py').read())

setup(
    name='quantum_electron',
    version=__version__,
    packages=find_packages(include=['quantum_electron']),
    install_requires=['numpy', 'scipy', 'matplotlib', 'shapely', 'scikit-image', 'pyvista', 'IPython', 'alive_progress'],
    extras_require={'test': ['pytest']}
)
