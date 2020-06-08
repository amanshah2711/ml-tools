import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="npml-amanshah2711", # Replace with your own username
    version="0.0.1",
    author="Aman Shah",
    author_email="amanshah2711@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
    ],
    python_requires='>=3.6',
)
