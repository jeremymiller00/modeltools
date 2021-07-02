import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="modeltools",
    version="0.0.1",
    author="Jeremy Miller",
    author_email="jeremymiller00@gmail.com",
    description="Tools to make modeling easier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/...",
    project_urls={
        "Bug Tracker": "https://github.com/.../issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)