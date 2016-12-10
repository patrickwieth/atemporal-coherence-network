from setuptools import setup

setup(
    name='acn',
    version='0.0.1',
    author='patrickwieth, neisok',
    author_email='patrick.wieth@googlemail.com',
    maintainer='',
    maintainer_email='',
    url='https://github.com/patrickwieth/atemporal-coherence-network',
    license="Wahrscheinlich",
    description='A atemporal coherence network, this means no time dependence in input patterns and connectrons learn coherence',
    long_description='yes you like ' +
                     'ne is ok',
    packages=['network', 'grid', 'util', 'tests'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering'
    ]
)
