from setuptools import setup, find_packages

setup(
    name='Adalpha',
    version='0.1',
    packages=find_packages(),
    url='http://github.com/DroneDude1/Max_AI',
    license='MIT',
    author='Max Clemetsen',
    author_email='maxedpc08@gmail.com',
    description='A Tensorflow optimizer designed to help avoid model collapse and improve generalization',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'tensorflow>=2.14',
        'pandas',
        'matplotlib'
    ],
    classifiers=[
        'Development Status :: 2 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache 2.0 License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',

    ],
)