import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

    name="sajeethphilip", # Replace with your username

    version="1.0.0",

    author="Ninan Sajeeth Philip",

    author_email="nsp@airis4d.com",

    description="CASSPER-v2 install",

    long_description=long_description,

    long_description_content_type="text/markdown",

    url="https://github.com/sajeethphilip/CASSPER2",

    packages=setuptools.find_packages(),

    classifiers=[

        "Programming Language :: Python :: 3",

        "License :: OSI Approved :: MIT License",

        "Operating System :: OS Independent",

    ],

    python_requires='>=3.6',

)
 
