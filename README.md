# Introduction to packaging #

Read the following to better understand packaging.

- [Official packaging tutorial](https://packaging.python.org/tutorials/packaging-projects/)
- [More in depth reference](https://packaging.python.org/guides/distributing-packages-using-setuptools/)
- [Same but from `setuptools` project itself](https://setuptools.readthedocs.io/en/latest/setuptools.html)

## Basics for starting a new repo ##
### Step 1: Change remote repo ###
- [ ] You must change the remote repository this is connected to.
   
        git remote rm origin
        git remote add origin <your new repo>
    Then make sure the changes took effect by running 

    
        $ git remote -v
        origin  https://github.com/imrsvdatalabs/<your new repo> (fetch)
        origin  https://github.com/imrsvdatalabs/<your new repo> (push)
 
- [ ] Fill in [`setup.py`](./setup.py) `setup()` call metadata.

### Step 2: Python setup ###
#### Python version ####
IMRSV uses [pyenv](https://github.com/pyenv/pyenv) for Python version management. Please set-up [pyenv](https://github.com/pyenv/pyenv) in your system by following [this tutorial](https://github.com/pyenv/pyenv#installation) (Please do not use the automatic install, do the Basic GitHub Checkout installation, Also install the dependencies if you haven't already done so before pyenv installation by following [this link](https://github.com/pyenv/pyenv/wiki/common-build-problems)) if you have not already done so.

- Set the Python version for your project by running the following command: 

```$ pyenv local <python-version>```

(e.g.: `pyenv local 3.7.2`). Pyenv manages python versions by maintaining a `.python-version` file. Refer to [this tutorial](https://github.com/pyenv/pyenv/blob/master/COMMANDS.md#pyenv-local) for more information about pyenv commands.

#### Python virtual environment ####
- Navigate to the project folder. (eg: `$ cd imrsv-production`)
- Create a new virtual environment named `venv` by: 
```$ python -m venv venv``` 
- Activate the virtual environment by:
```$ . ./venv/bin/activate```

## Documentation ##
IMRSV uses [sphinx](http://www.sphinx-doc.org/en/master/) for package documentation. All sphinx related files should be inside the [docs](./docs) folder.

Please refer to [guides/sphinx_guide.md](./guides/sphinx_guide.md) for further details about sphinx documentation and standard practices recommended within IMRSV

## Once you've written code ##
- [ ] Write basic [*smoke* tests], that at least test that:
    - [ ] Your code can be imported.
    - [ ] Then replace with basic smoke tests of your data loading.
    - [ ] Then replace with basic smoke tests of your top-level functions, classes, etc.
- [ ] Add [type annotations](https://docs.python.org/3/library/typing.html) to your user-facing code (your public APIs). This will be checked by mypy.
- [ ] Add code [documentation](https://devguide.python.org/documenting/). Documentation for functions should be kept seperate from general use documentation like 'getting started' guides.

## When you're ready to push ##
- [ ] Get a code review from both an applicable researcher and a good programmer.
- [ ] `pip freeze > requirements.txt` in a virtual environment with ONLY runtime (excluding build-time, training-time, test-time, experimentation) dependencies. Please don't edit `requirements.txt` unless you absolutely have to.
- [ ] The `setup.py` file is configured to use git based versioning of releases (This is done by adding `use_scm_version=True` and `setup_requires=['setuptools_scm']` into the [setup.py](../setup.py) file ).
    - A git tag can be added as follows:
    ```
    git tag -a v0.0.1 -m "v0.0.1, first release to prod"
    git push origin --tags
    ```
    - Please follow [this tutorial](https://drvnintelligence.com/setting-up-a-pip-installable-python-3-git-repo/) to learn more about git tags.
- [ ] You can test your package locally before pushing it to github by using commands such as `pip install -e .` ([More details here](https://pip.pypa.io/en/stable/reference/pip_install/#options)) or `python setup.py develop` ([more details here](https://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install))

### A note about `requirements.txt` file and experimental code:
 Please don't add requirements from experimental code into the `requirements.txt` file. This file is meant to have the requirements for the final production code. Requirements for experimental code can be put inside subfolders containing the requirements.

 Eg:
  `Jupyter notebook` experiment on pdftotext: Add this inside `experiments/notebooks/pdftotext`. Add a `requirements.txt` or a plain text file containing information about libraries/packages required for this experiment along with any special instructions on installing them.


## Files in this template ##

- `src`; because it keeps the installed modules separate from the mess of other
  files. Also forces you to `pip install` your code, or the tests won't run.
- `setup.py`; package metadata go here
- `.env-sample`; add any *sample*, environment variables here as required.
  Secrets do not belong in here. Leave only placeholders if the case.
- `.gitignore`
- .`python-version`; Python versions should be frozen, just like packages.
- `bin`; put scripts to be installed in the `$PATH` here.
- `requirements.txt`; `pip freeze` into here, per guidelines.
- `scripts`; put developer scripts not to be installed in `$PATH` here.
- `tests`; [`pytest`] style

### Preconfigured files - Do not modify ###

- `.coveragerc`; Configuration of [`coverage`], measuring which lines of code
  were tested.
- `.flake8`; Style and linter config.
- `.rc`; Loads `.env`.
- `.travis.yml`; Tells hosted CI how to run tests automatically.
- `MANIFEST.in`; Ensures non-Python files in [`src`](./src) end up in package
  too.
- `mypy.ini`; Configuration of type checker.
- `pytest.ini`; Configuration of test runner.


## A note on Pipenv ##

In @pilona's experience, Pipenv has been slow, awkward to use (some use cases
not fully fleshed out). Don't bother. Wait for more mature tools.
[`requirements.txt`](./requirements.txt) and [`setup.py`](./setup.py).

Pipenv provides separation of runtime and dev, and trivial regeneration of the dump of only the former, is one of the value-adds of Pipenv. Too bad it's overall not worth it.

## TODO for template developers ##

- [ ] Maybe make `pydoc imrsv` work? Seems like we'd actually have to define
      the `imrsv` module (e.g., `__init__.py`, etc.).

[*smoke* test]: https://en.wikipedia.org/wiki/Smoke_testing_(software)
[type annotations]: https://docs.python.org/3/library/typing.html
[`pytest`]: https://docs.pytest.org/en/latest/
