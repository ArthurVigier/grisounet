from pathlib import Path

from setuptools import find_packages
from setuptools import setup


def parse_requirements(path: str) -> list[str]:
    lines = Path(path).read_text().splitlines()
    requirements = []
    for line in lines:
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith(("-r", "-c")):
            continue
        if "git+" in entry:
            continue
        requirements.append(entry)
    return requirements


setup(
    name="grisounet",
    version="0.0.1",
    description="Detection de taux de concentration de PM2.5 à partir de données de séries temporelles",
    license="MIT",
    author="Multiple",
    # url="https://github.com/lewagon/taxi-fare",
    install_requires=parse_requirements("requirements/app.txt"),
    packages=find_packages(),
    # include_package_data: to install data from MANIFEST.in
    include_package_data=True,
    zip_safe=False,
)
