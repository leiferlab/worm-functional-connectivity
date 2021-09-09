#!/usr/bin/env python

import os
from distutils.command.build_ext import build_ext
from distutils.core import Extension, setup


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)


_integration = Extension(
    "wormfunconn._integration",
    sources=["wormfunconn/_integration.cpp"],
    extra_compile_args=["-O3"],
)


_convolution = Extension(
    "wormfunconn._convolution",
    sources=["wormfunconn/_convolution.cpp", "wormfunconn/convolution.cpp"],
    extra_compile_args=["-O3"],
)


setup(
    cmdclass={"build_ext": CustomBuildExtCommand},
    name="wormfunconn",
    version=get_version("wormfunconn/__init__.py"),
    description="Functional connectivity atlas for the C. elegans brain",
    author="Francesco Randi",
    author_email="francesco.randi@gmail.com",
    packages=[
        "wormfunconn",
    ],
    ext_modules=[
        _integration,
        _convolution,
    ],
    install_requires=["numpy", "matplotlib"],
    package_data={"wormfunconn": ["aconnectome_ids.txt"]},
    entry_points={
        "console_scripts": [
            "use_mock_atlas_scalar=scripts.use_mock_atlas_scalar:main",
        ]
    },
)
