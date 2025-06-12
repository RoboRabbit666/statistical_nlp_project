"""Setup script for the Advanced NLP Project."""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Advanced Natural Language Processing Pipeline"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="advanced-nlp-project",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced Natural Language Processing Pipeline with RAG, Keyword Extraction, and Sentence Ranking",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/advanced-nlp-project",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nlp-project=nlp_project.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="nlp, rag, keyword-extraction, sentence-ranking, fact-checking, transformers",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/advanced-nlp-project/issues",
        "Source": "https://github.com/yourusername/advanced-nlp-project",
        "Documentation": "https://advanced-nlp-project.readthedocs.io/",
    },
)