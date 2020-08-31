import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()


setuptools.setup(
    name='nba-win-probability',
    version="0.0.1",
    author="Brent Satterwhite",
    author_email="bsatterwhite@gmail.com",
    description="NBA Win Probability model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bsatterwhite3/nba-win-probability",
    packages=setuptools.find_packages(),
    # install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.7',
)
