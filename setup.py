from setuptools import setup, find_packages

setup(
    name="torchseis",
    version="0.0.1",
    author="Jintao Li",
    description="", 
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JintaoLee-Roger/seistorch",
    packages=find_packages(),
    # install_requires=open(
    #     "requirements.txt").read().splitlines(),
    install_requires=['torch'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8', 
)
