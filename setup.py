import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

    name="CASSPER2", # Replace with your username

    version="2.3.1",

    author="Ninan Sajeeth Philip",

    author_email="nsp@airis4d.com",

    description="CASSPER-v2 install",

    long_description=long_description,

    long_description_content_type="text/markdown",

    url="https://github.com/sajeethphilip/CASSPER2/",

    packages=setuptools.find_packages(include=['CASSPER2.*','CASSPER']),
    install_requires=[
        'CASSPER2.py',
        'argparse',
        'image-slicer',
        'imageio',
        'importlib-metadata',
        'imutils',
        'matplotlib',
        'mrcfile',
        'numpy',
        'opencv-python',
        'Pillow',
        'ray',
        'requests',
        'scikit-image',
        'scikit-learn',
        'scipy',
        'image_slicer',
        'sklearn',
        'tensorboard',
        'tensorflow',
        'tf_slim'
        ],

    classifiers=[

        "Programming Language :: Python :: 3",

        "License :: OSI Approved :: MIT License",

        "Operating System :: OS Independent",

    ],

    python_requires='>=3.6',

)
 
