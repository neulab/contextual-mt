import setuptools

setuptools.setup(
    name="dialogue-mt",
    version="0.0.1",
    author="Patrick Fernandes, Kayo Yin, Kervy Dante",
    author_email="pfernand@cs.cmu.edu",
    description="Package containing experiments related to Dialogue Translation",
    url="https://github.com/neulab/dialogue-mt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    entry_points={"console_scripts": ["dialogue-train = dialogue_mt.train:cli_main", "dialogue-evaluate = dialogue_mt.evaluate:main", "contrastive-evaluate = dialogue_mt.contrastive:main"]},
    python_requires=">=3.6",
)
