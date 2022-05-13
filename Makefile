setup:
	@echo "Installing Updates ... "
	sudo apt update
	sudo apt-get update
	@echo "Installing PIP ... "
	sudo apt install python3-pip
	@echo "Installing Python Dependencies ... "
	pip3 install numpy
	pip3 install pandas
	pip3 install matplotlib
	pip3 install sklearn
	pip3 install tqdm
	
clean:
	rm -rf __pycache__
	
	
