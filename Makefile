data = false

# Downloads data from google drive
download-data:
	@echo "Downloading data from Google Drive..."
	python download_data.py
	@echo "Data has been downloaded"

# Loads the needed module on Euler
load-module:
	@echo "Loading module..."
	module load gcc/8.2.0 python_gpu/3.10.4
	@echo "Python 3.10.4 with GPU has been loaded"

# Uploads the code without data to Euler
copy-to-euler:
	@echo "Copying files to Euler..."
	mv Dataset.zip ../ || true
	ssh $(username)@euler.ethz.ch "rm -rf /cluster/scratch/$(username)/AI4Good/.git/"
	scp -r . $(username)@euler.ethz.ch:/cluster/scratch/$(username)/AI4Good/
	mv ../Dataset.zip . || true
	@echo "Files have been copied to Euler"

# Uploads the data to Euler
upload-data:
	@echo "Uploading data to Euler..."
	scp -r ./Dataset.zip $(username)@euler.ethz.ch:/cluster/scratch/$(username)/AI4Good/
	@echo "Data uploaded to Euler"

# Unzips the data
unzip-data:
	@echo "Unzipping data..."
	unzip Dataset.zip
	mv Dataset dataset
	@echo "Data unzipped to ./dataset"

# Sets up the pipenv environment
set-pipenv:
	@echo "Installing new packages..."
	pipenv install -r requirements.txt
	@echo "Packages have been installed"
	@echo "Setting up pipenv shell..."
	pipenv shell
	@echo "Pipenv shell has been set up"

# Outputs the installed packages to requirements.txt
freeze:
	@echo "Freezing packages..."
	pip freeze > requirements.txt
	@echo "Packages have been frozen"

# Exits from pipenv
exit-pipenv:
	@echo "Exiting pipenv shell..."
	exit
	@echo "Pipenv shell has been exited"

# Runs the code
run:
	@echo "To be discussed"
	python main.py

# Runs the code on Euler, still has to be discussed
run-euler:
	@echo "Running code on Euler..."
	bsub python main.py
	@echo "Code has been run on Euler"

# In order to run the code locally just run the following command:
# make pipeline data=true username=your_username
# If you already have the data, just run the following command:
# make pipeline data=false username=your_username
pipeline:
	@echo "Running pipeline..."
	if [$(data) = true]; then\
		make download-data;
		make unzip-data;
	fi
	make set-pipenv
	make run
	@echo "Pipeline has been run"

# In order to run the code on Euler just run the following command locally:
# make pipeline-local data=true/false username=your_username
# Now go to Euler and see the comment for pipeline-euler
pipeline-local:
	@echo "Running the local part of pipeline..."
	if $(data) = true; then\
		make download-data;
		make upload-data;
	fi
	make copy-to-euler
	@echo "Local part of pipeline has been run"

# After the local part of the pipeline has been run, go to Euler and run the following command:
# make pipeline-euler data=true/false username=your_username
pipeline-euler:
	@echo "Running the Euler part of pipeline..."
	make load-module
	if $(data) = true; then\
		make unzip-data;
	fi
	make set-pipenv
	make run-euler
	@echo "Euler part of pipeline has been run"

# Used only for testing
test:
	if $(data) = true; then\
		echo "Data is true";\
	else\
		echo "Data is false";\
	fi