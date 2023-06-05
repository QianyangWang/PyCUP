from setuptools import setup, find_packages

with open(r"README.md",encoding="utf-8") as f:
  long_description = f.read()


setup(
    name='pycup',
  version='0.1.7',
  description='An auto-calibration tool for environmental models based on heuristic algorithms and uncertainty estimation theory.',
  long_description=long_description,
  long_description_content_type="text/markdown",
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
  install_requires=['numpy', 'matplotlib','scipy','pyDOE','statsmodels','pandas','prettytable'],
  python_requires='>=3'
  )