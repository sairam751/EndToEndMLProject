#Setup .py file is used build our application as a package
from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        # HYPEN_E_DOT value in requirements.txt will run the setup.py file also in order to remove that we are removing this
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


#Metadata of our project
setup(
name = 'mlproject',
version='0.0.1',
author='Sai Sri Ram Reddy',
author_email='sairam751348@gmail.com',
packages=find_packages(),
install_requires = get_requirements('requirements.txt')


)