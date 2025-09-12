import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="specbox",
    version="0.1.1",
    author="Yuming Fu",
    author_email="fuympku@outlook.com",
    description="specbox - A simple tool to manipulate and visualize optical spectra for astronomical research.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rudolffu/specbox",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "astropy",
        "pyqtgraph",
        "PySide6",
        "specutils",
        "matplotlib",
        "pandas",
        "requests",
        "pillow",
        "astroquery",
    ],
    package_data={
        'specbox': ['data/*'],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE V3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
