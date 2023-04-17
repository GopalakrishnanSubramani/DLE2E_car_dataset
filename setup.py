from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT= '-e .'
def get_requirements(file_path:str)->List[str]:
    """Returns a list of requirements"""

    requirements=[]
    with open(file_path, 'r') as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.replace("/n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements



setup(
    name='Deep Learning Project',
    version='0.0.1',
    author='Gopalakrishnan',
    author_email='gopalinlatvia@gmail.com',
    packages=find_packages(),
    install_requires =get_requirements('requirements.txt'),
)
