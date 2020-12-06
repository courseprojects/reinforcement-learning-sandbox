from setuptools import setup

setup(
    name='roboray',
    version='0.1.0',    
    description='Package for running robosuite with ray',
    url='https://github.com/peterdavidfagan/CS221-Project'
    author='Peter David Fagan',
    author_email='pfagan@stanford.edu',
    license='MIT',
    packages=['roboray'],
    python_requires='>=3.5',
    install_requires=[
    	glfw>=1.4.0
		numpy>=1.13.3
		Cython>=0.27.2
		imageio>=2.1.2
		cffi>=1.10
		fasteners~=0.15
		jupyter
		torch
		robosuite
		wandb
    ]
    entry_points = {
    	'console_scripts':['roboray-train=roboray.cli:main']
    }
)