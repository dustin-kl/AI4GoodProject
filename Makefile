download-data:
	@echo "Downloading data..."
	wget -r -np -k https://portal.nersc.gov/project/ClimateNet/climatenet_new/
	mv portal.nersc.gov/project/ClimateNet/climatenet_new/train/data* dataset/
	mv portal.nersc.gov/project/ClimateNet/climatenet_new/test/data* dataset/
	rm -rf portal.nersc.gov
	@echo "Data has been downloaded"

create-env:
	@echo "Creating environment..."
	pipenv install -r requirements.txt
	pipenv shell
	@echo "Environment has been created"

test-run:
	@echo "Running with small dataset..."
	python main.py --small_dataset

run:
	@echo "Running code..."
	python main.py

exit-env:
	exit

remove-env:
	pipenv --rm
