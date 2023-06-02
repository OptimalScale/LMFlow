import os
from setuptools import find_packages
from setuptools import setup
import subprocess

folder = os.path.dirname(__file__)
version_path = os.path.join(folder, "src", "lmflow", "version.py")

__version__ = None
with open(version_path) as f:
  exec(f.read(), globals())

req_path = os.path.join(folder, "requirements.txt")
install_requires = []
if os.path.exists(req_path):
  with open(req_path) as fp:
    install_requires = [line.strip() for line in fp]

readme_path = os.path.join(folder, "README.md")
readme_contents = ""
if os.path.exists(readme_path):
  with open(readme_path, encoding='utf-8') as fp:
    readme_contents = fp.read().strip()

setup(
    name="lmflow-finetune",
    version=__version__,
    description="LMFlow: Large Model Flow.",
    author="The LMFlow Team",
    long_description=readme_contents,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={},
    install_requires=install_requires,
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    requires_python=">=3.9",
)

# Must be called after all dependency installed, since flash-attn setup.py
# relies on torch, packaging, etc.
try:
  gpu_state = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
  if b"A100" in gpu_state:
    subprocess.call(["pip", "install", "flash-attn==1.0.4"])
except:
  pass
