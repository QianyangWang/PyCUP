from setuptools import setup, find_packages

setup(
    name='pycup',
  version='0.1.2',
  description='An auto-calibration tool for environmental models based on heuristic algorithms and uncertainty estimation theory.',
  author='Qianyang Wang',
  author_email='wqy07010944@hotmail.com',
  url='https://github.com/QianyangWang/PyCUP',
  license='MIT License',
  keywords='optimization',
  project_urls={
   'Documentation': 'https://github.com/QianyangWang/PyCUP/DOCUMENT',
   'Source': 'https://github.com/QianyangWang/PyCUP/pycup',
  },
  packages=find_packages(exclude=["test", "test.*"]),
  package_data={'pycup':['document/*.pdf']},
  install_requires=['numpy', 'matplotlib','scipy','pyDOE','statsmodels'],
  python_requires='>=3'
  )