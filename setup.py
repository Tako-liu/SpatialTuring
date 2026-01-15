from setuptools import setup, find_packages

setup(
    name="spatialturing",
    version="0.1.0",
    description="PyTorch-accelerated framework for unmasking Turing reaction-diffusion logic in spatial transcriptomics.",
    author="Tako-liu",  # 你的 GitHub 用户名
    # author_email="your.email@example.com", # 选填
    url="https://github.com/Tako-liu/SpatialTuring",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "scipy>=1.7",
        "scanpy>=1.8",
        "torch>=1.10",  # 核心依赖
        "tqdm",         # 进度条
        "matplotlib",
        "seaborn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
