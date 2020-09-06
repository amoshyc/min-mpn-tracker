tracktor_detections: 
	cd tracktor_detections
	wget https://vision.in.tum.de/webshare/u/brasoand/mot_neural_solver/prepr_det_files.zip
	unzip prepr_det_files.zip
	cd ..

reid_weights:
	mkdir -p weights
	cd weights
	wget https://vision.in.tum.de/webshare/u/brasoand/mot_neural_solver/resnet50_market_cuhk_duke.tar-232
	cd ..

setup: reid_weights tracktor_detections
	python prepare_mot15.py
	python prepare_mot17.py

train:
	python main.py