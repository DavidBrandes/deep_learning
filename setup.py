from setuptools import setup, find_packages

setup(
    name='dl',
    version='0.1.0',
    description='',
    author='',
    author_email='',
    url='',
    packages=find_packages(include=['dl', 'dl.*']),
    install_requires=[
        'kornia==0.7.0',
        'matplotlib==3.7.1',
        'numpy==1.25.2',
        'pillow==9.4.0',
        'torch==2.0.1',
        'torchvision==0.15.2'
    ],
)