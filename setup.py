import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="specbox",
    version="0.1.0-alpha",
    author="Yuming Fu",
    author_email="fuympku@outlook.com",
    description=" ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rudolffu/specbox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE V3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)