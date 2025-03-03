"""Setup configuration for jira-ai package."""
from setuptools import setup, find_packages

setup(
    name="jira-ai",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "jira",
        "openai",
        "pyarrow",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
        ],
    },
    python_requires=">=3.10",
)
