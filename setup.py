import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='GRMF',  
     version='0.1',
     scripts=['GRMF'] ,
     author="Anonymous Author",
     author_email="anonymous@anonymous.com",
     description="Robust Matrix Factorization with Grouping Effect",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/GroupingEffects/GRMF",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
     ],
 )
