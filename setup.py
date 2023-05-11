from setuptools import find_packages,setup
from typing import List

Hypen_dot="-e ."

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as fileobj:
        requirements=fileobj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if(Hypen_dot in requirements):
            requirements.remove(Hypen_dot)

        return requirements



setup(
    name='DiamondPricePrediction',
    version='0.0.1',
    author='Mari',
    author_email='maripvm14@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)