from setuptools import setup, find_packages

setup(
    name="genre_classification_tf",
    version="0.4.3",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow",
        "librosa",
        "matplotlib",
        "scikit-learn",
        "customtkinter",
    ],
    description="A TensorFlow project for genre classification of music",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cinaaaaaaaaar/MusicGenreClassifierf",
    python_requires=">=3.9",
)
