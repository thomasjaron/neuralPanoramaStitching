# ------------------------------------------------------
# VARIABLES
# ------------------------------------------------------
VENV=. venv/bin/activate
VENV_WINDOWS=call venv/Scripts/activate
# ------------------------------------------------------
# MAIN COMMANDS
# ------------------------------------------------------
venv-python-windows:						## Create a new Python virtualenv in venv folder
	python -m venv venv
	$(VENV_WINDOWS) && \
		python -m pip install pip-tools

venv-python:
	python3 -m venv venv
	$(VENV) && \
		python -m pip install pip-tools

install-deps-windows: 						## Install all (including dev) requirements in virtualenv
	$(VENV_WINDOWS) && \
		python -m pip install --upgrade pip && \
		pip-compile --output-file ./requirements.txt ./requirements.in && \
		pip-compile --output-file requirements-dev.txt requirements-dev.in && \
		pip install -r requirements.txt  -r requirements-dev.txt

install-deps:
	$(VENV) && \
		python -m pip install --upgrade pip && \
		pip-compile --output-file ./requirements.txt ./requirements.in && \
		pip-compile --output-file requirements-dev.txt requirements-dev.in && \
		pip install -r requirements.txt  -r requirements-dev.txt

install-cuda-deps-windows:
	$(VENV_WINDOWS) && \
		python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

install-cuda-deps:
	$(VENV) && \
		python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

upgrade-deps-windows: 						## Install and upgrade all requirements in virtualenv
	$(VENV_WINDOWS) && \
		del requirements.txt requirements-dev.txt && \
		python -m pip install --upgrade pip && \
		pip-compile --upgrade --output-file ./requirements.txt ./requirements.in && \
		pip-compile --upgrade --output-file ./requirements-dev.txt ./requirements-dev.in && \
		python -m pip install --upgrade -r torchcuda.txt -r ./requirements.txt  -r ./requirements-dev.txt

upgrade-deps:
	$(VENV) && \
		rm -f requirements.txt requirements-dev.txt && \
		python -m pip install --upgrade pip && \
		pip-compile --upgrade --output-file ./requirements.txt ./requirements.in && \
		pip-compile --upgrade --output-file ./requirements-dev.txt ./requirements-dev.in && \
		python -m pip install --upgrade -r torchcuda.txt -r ./requirements.txt  -r ./requirements-dev.txt
