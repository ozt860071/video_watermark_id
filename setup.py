from setuptools import setup, find_packages

setup(
    name="video_watermark",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "av",               # PyAV — FFmpeg bindings
        "numpy",
        "scipy",
        "scikit-image",
        "Pillow",
        "tqdm",
        "matplotlib",
        "bchlib",
        # trustmark is optional — install separately:
        #   pip install trustmark
    ],
    extras_require={
        "ml": ["trustmark"],
        "gpu": ["torch", "torchvision"],
    },
)
