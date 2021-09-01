from setuptools import setup

setup(
    name='draw',
    version='0.1.0',
    py_modules=["server"],
    install_requires=[
        "numpy==1.21.2",
        "opencv-python==4.5.3.56",
        "pyserial==3.5",
        "tqdm==4.62.2",
        "click==8.0.1",
    ],
    entry_points={
        'console_scripts': [
            'draw = server:cli',
        ],
    },
)