import setuptools

with open("README.md", 'r') as fp:
	long_description = fp.read()

setuptools.setup(
	name = "rhodes",
	version = "1.0.2",
	author="Andrew Durden, Abolfazl Farahani, Saed Rezayi",
	author_email="durden4th@gmail.com, a-farahani@uga.edu, saedr@uga.edu",
	license='MIT',
	description="A package for neuron detection.",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/dsp-uga/team-rhodes-P3",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
	],
	test_suite='nose.collector',
	tests_require=['nose'],
    install_requires=['imageio', 'thunder-extraction', 'joblib', 'image_slicer'],
)
