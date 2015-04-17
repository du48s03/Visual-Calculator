# Visual-Calculator

Requires:

open CV
scikit-learn

To generate the training dataset, use bin/get_training_data.py. Run like this:

python get_training_data.py <dataset_file_name>

A window will appear. Press 'z' to take a picture, and enter the label of this image in the counsel. If you want to discard ths image because the quality of the image is not good, simply press enter without typing any label. Then repeat the same process until you are satisfied with the number of data. Press 'q' to quite the program. A file called containing the training dataset will appear. 

To see the training data, use bin/show_dataset.py. Usage:

python show_dataset.py <dataset_file_name>

To train the program, use bin/train.py. Usage:

python train.py <dataset_file_name> <model_file_name>

The program trains on the specified data, and generates a model file with the specified name. The current training model is nearest neighbors. 

To use the program to do classification, use bin/main.py. Usage:

python main.py <model_file_name>

A window will apear. Make a pose and press 'z'. The result of the classification will be shown in the counsel. 
