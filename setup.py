from setuptools 
import setup, find_packages
import io

setup(
    name='TEFL-mRNA',
    version='0.1.0',
    author='Siqi Wang',
    author_email='siqiwang@ucla.edu',
    description='TEFL-mRNA: a hybrid deep-learning model integrating CNN and RNN architectures to predict the Translation Efficiency (TE) from Full-Length mRNA sequences.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your-repo-name',
    packages=['TEFL-mRNA'], 
    install_requires=[
        'gpu': 'tensorflow==2.12.0',
        'keras==2.12.0',
        'pandas==2.2.3',
        'numpy==1.23.5'
    ],
    python_requires='>=3.7',
)