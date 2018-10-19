# Generative Adversarial Networks(GAN) with MINE: Mutual Information Neural Estimation

## Environment setup & dependency installation
```
git clone https://github.com/rikenmehta03/gan_mine.git
cd gan_mine
./install
```
This commands will setup a virtual environment using python3 and install the required packages in the same environment. Install script will also create an alias for activating the virtual environment: `cv_env`

## Repository structure
Every module will be a directory containing `__init__.py` file and other subfiles. Below are the initial modules we need to write. 
- **gan** : This module contains `GanTrainer` class in `gan_trainer.py` file
- **utils** : This module will contains all the utility functions or classes we write. For example `Logger` class in `logger.py` file.
- **model** : This module will contain all the different CNN architecure we try for discriminator or generator. Try to keep discriminator and generator classes in separate files. 

We will add  `__init__.py` file in every module and import all the class of that module (See `/gan/__init__.py` for refrence), so that we can directly import them in `train.py` or `evaluate.py` like this: 
```
from gan import GanTrainer
```

## Naming conventions & style guide
- use `CapitalCase` for class names
- use `smallcase` for module names
- use `snake_case` for variable and file names. 
- Use sensible names for variables. Do not use variable names like `c`, `i`, `p` unless untill those are very trivial. 

## Contributing to git repository
- Make sure to pull the latest code before you start working on anything to avoid headache of merge conflicts.  
- Add proper commit message. One practice I saw was to add commit type along with the commit message like this: 
    - `[UPDATE] Modify so and so class in module foo`
    - `[ADD] Created so and so class in module foo`
    - `[FIX] NoneType bug fix in so and so class`
