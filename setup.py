#!/usr/bin/env python

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):

        # Import numpy here, only when headers are needed
        import numpy
        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())
        
        # Call original build_ext command
        build_ext.run(self)

_integration = Extension('wormfunconn._integration',
                    sources = ['wormfunconn/_integration.cpp'],
                    extra_compile_args=['-O3'])
                    
_convolution = Extension('wormfunconn._convolution',
                    sources = ['wormfunconn/_convolution.cpp',
                               'wormfunconn/convolution.cpp'],
                    extra_compile_args=['-O3'])

                    
setup(name='wormfunconn',
      version='0.1',
      description='Functional connectivity atlas for the C. elegans brain',
      author='Francesco Randi',
      author_email='francesco.randi@gmail.com',
      packages=['wormfunconn',],
      ext_modules = [_integration,_convolution,],
      package_data={'wormfunconn': ['aconnectome_ids.txt']}
     )
