from setuptools import setup, find_packages

setup(
    name='newme',
    version='0.3.0',
    description='A Flask web app with SQL-backed setup flow',
    url='https://github.com/gu-wmn/webapp',
    author='Kaj Ailomaa',
    author_email='kaj.ailomaa@gu.se',
    license='MIT',
    packages=find_packages(where="src", include=["newme", "newme.*"]),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "newme": ["data/wmn_annotations.json"],
    },
    install_requires=[
        'flask',
        'flask-sqlalchemy',
        'gunicorn',
        'requests'
    ],
    python_requires='>=3.12',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT No Attribution License (MIT-0)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.12',
    ],
)
