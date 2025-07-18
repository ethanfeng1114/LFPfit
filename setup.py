from setuptools import setup

setup(
    name='LFPfit',
    version='0.1.0',    
    description='LFPfit-- temperature dependent LFP OCP',
    url='',
    author='Archie Mingze Yao',
    author_email='amyao@umich.edu',
    license='MIT License',
    packages=['LFPfit'],
    install_requires=[
                      'numpy==1.24',       
                      'torch==2.0.0',
                      'pandas',
                      'scipy',
                      'matplotlib',         
                      ],

)

