from setuptools import setup, find_packages

setup(
    name='Not just semantics - Web app',
    version='0.1.0',
    description='A web app that applies standoff annotation to corpora',
    url='https://github.com/gu-wmn/webapp',
    author='Kaj Ailomaa',
    author_email='kaj.ailomaa@gu.se',
    license='MIT',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        'regex',
        'flask',
        'requests',
        'gunicorn'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT No Attribution License (MIT-0)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.12',
    ],
)
