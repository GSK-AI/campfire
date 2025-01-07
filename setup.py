import os

from setuptools import find_packages, setup

with open("VERSION") as f:
    version = f.read().strip()

if os.getenv("PRERELEASE"):
    version += os.getenv("PRERELEASE")
    with open("VERSION", "w") as f:
        f.write(f"{version}\n")

setup(
    name="channel_agnostic_vit",
    version=version,
    description="",
    packages=find_packages(),
)