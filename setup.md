# here we as a group will discuss setup instructions for the project
to add more requirements then do:
```bash
pip freeze > requirements.txt
```
to install requirements:
```bash
pip install -r requirements.txt
```
make sure to do
```bash
pre-commit install
```
haris
this is how I run my venv

.\.venv\Scripts\activate
maham
this is how I run my venv

.\mlops-venv\Scripts\activate



pre-commit
      # - id: ruff-check
      #   name: ruff (check + fix)
      #   entry: ruff check --fix
      #   language: system
      #   types: [python]