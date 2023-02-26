from setuptools import setup, find_packages

setup(
  name = 'simple-toolformer',
  packages = find_packages('src'),
  package_dir={'':'src'},
  version = '0.0.1',
  license='MIT',
  description = 'Toolformer',
  long_description_content_type = 'text/markdown',
  author = 'mrcabbage972',
  url = 'https://github.com/mrcabbage972/simple-toolformer',
  keywords = [
    'deep learning',
    'transformers',
    'natural language processing'
  ],
  install_requires=[
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
