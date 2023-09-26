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
        'kornia',
        'matplotlib',
        'numpy',
        'pandas',
        'pillow',
	'tensorboard'
        'torch',
        'torchvision',
	'tqdm',
	'scipy'
    ],
)
