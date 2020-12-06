from setuptools import setup

setup(
    name='roboray',
    version='0.1.0',    
    description='Package for running robosuite with ray',
    author='Peter David Fagan',
    author_email='pfagan@stanford.edu',
    license='MIT',
    packages=['roboray'],
    python_requires='>=3.5',
    entry_points = {
    	'console_scripts':['roboray-train=roboray.cli:main']
    }
)