from setuptools import setup, find_packages

setup(
    name='eeg-hyena',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'mne',
        'numpy',
        'scikit-learn',
    ],
    author='NullLabTests',
    author_email='your.email@example.com',
    description='EEG to text using adapted Hyena Hierarchy',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/NullLabTests/EEG-Hyena',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
