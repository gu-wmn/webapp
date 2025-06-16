from setuptools import setup, find_packages

setup(
    name='newme',
    version='0.2.0',
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
