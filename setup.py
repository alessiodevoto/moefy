from setuptools import setup

setup(
    name='moefy',
    url='https://github.com/alessiodevoto/moefy.git',
    author='Alessio Devoto',
    packages=['moefy'],
    install_requires=[], # TODO add here torch geometric!
    version='0.1',
    # The license can be anything you like
    # license='MIT',
    description='Make Mixtures of Experts out of (almost) anything!',
    long_description=open('README.md').read(),
)