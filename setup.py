# setup.py
from setuptools import setup, find_packages

setup(
    name="human_in_the_loop_transcribe",
    version="0.1.0",
    description="Human-in-the-loop transcription workflow",  
    author="Haruka Kataoka",
    author_email="123.popopcorn@gmail.com",
    url="https://github.com/khkata/human-in-the-loop-transcribe",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "jiwer",
        "fugashi[unidic-lite]",
        "tqdm",
        "soundfile",
        "torchaudio",
        "transformers",
        "datasets",
        "scipy",
        "pandas",
        "scikit-learn",
        "neologdn",
        "mojimoji",
        "optuna",
        "pydub",
        "ipywidgets",
    ],
)