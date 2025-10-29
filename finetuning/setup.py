import pathlib

from setuptools import find_packages, setup


def read_requirements():
    """Read requirements from requirements.txt if it exists."""
    here = pathlib.Path(__file__).parent
    req_file = here / "requirements.txt"
    if not req_file.exists():
        return []
    lines = req_file.read_text(encoding="utf-8").splitlines()
    requirements = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


setup(
    name="rex-omni-finetuning",
    version="0.0.1",
    author="Mountchicken",
    author_email="mountchicken@outlook.com",
    url="https://github.com/IDEA-Research/Rex-Omni",
    description="Rex-Omni finetuning utilities",
    packages=find_packages(exclude=("tests", "examples")),
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.8",
)
