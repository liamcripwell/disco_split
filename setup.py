from setuptools import setup, find_packages

setup(
    name="disco_split",
    version="0.0.1",
    author="Liam Cripwell",
    author_email="liam.cripwell@loria.fr",
    description="Code for running discourse-based sentence splitting experiments.",
    packages=find_packages(),
    include_package_data=True,
)
