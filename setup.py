from distutils.core import setup

setup(
    name='comfy-bayes',
    version='0.0.1',
    packages=['conversion_test'],
    url='https://github.com/eki107/comfy-bayes',
    license='',
    author='ssoos',
    author_email='ssoos@pixelfederation.com',
    description='comfy bayesion a/b test tools for lazy analysts',
    requires=['numpy', 'scipy']
)