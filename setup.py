from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='peters_rl_sandbox',
    version='0.1.0',    
    description='Package for testing reinforcement learning algorithms',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/peterdavidfagan/reinforcement_learning_sandbox.git',
    author='Peter David Fagan',
    author_email='peterdavidfagan@gmail.com',
    license='MIT',
    python_requires='>=3.5',
    install_requires=[
    	'glfw>=1.4.0',
		'numpy>=1.13.3',
		'Cython>=0.27.2',
		'imageio>=2.1.2',
		'cffi>=1.10',
		'fasteners~=0.15',
		'jupyter',
		'torch',
		'robosuite',
		'wandb',
        'pyyaml'
    ],
    entry_points = {
    	'console_scripts':['sandbox=sandbox.cli.cli_temp:main']
    }
)