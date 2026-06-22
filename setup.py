from setuptools import setup

# The repo directory is `video_watermark_id`, but internal imports use the
# package name `video_watermark`. Map the four subpackages explicitly so
# `pip install -e .` exposes them under that name.
_SUBPACKAGES = ["benchmark", "dct", "trustmark_video", "utils"]

setup(
    name="video_watermark",
    version="0.1.0",
    packages=["video_watermark"] + [f"video_watermark.{p}" for p in _SUBPACKAGES],
    package_dir={"video_watermark": "."} | {f"video_watermark.{p}": p for p in _SUBPACKAGES},
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
