from setuptools import find_packages, setup

setup(
    name="isaac-go2-low-level-rl",
    version="0.1.0",
    description="Custom self-trained Unitree Go2 low-level locomotion task for Isaac Lab.",
    packages=find_packages(),
    python_requires=">=3.10",
    zip_safe=False,
)
