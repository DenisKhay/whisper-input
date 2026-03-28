from setuptools import setup, find_packages

setup(
    name="whisper-input",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "faster-whisper",
        "sounddevice",
        "numpy",
        "pynput",
        "pystray",
        "Pillow",
        "PyYAML",
    ],
    entry_points={
        "console_scripts": [
            "whisper-input=whisper_input.main:main",
        ],
    },
)
