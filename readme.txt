This folder contains the implementation of MuGACD model on the dataset CESMARD as contribution to the paper titled "Aspect-based Complaint and Cause Detection: A Multimodal Generative Framework with External Knowledge Infusion".


To run this experiment-

REQUIREMENTS-

Some major libraries are-

torchvision - 0.14.1
torch - 1.13.1
numpy - 1.21.5
scikit-learn - 1.0.2
transformers - 4.25.1

PATH-
-In the exp_model.py file, to load the dataset correctly, Change the path of of files in code. The relevant names of paths in the code are -  "path_to_images", "path_to_train", "path_to_val", "path_to_test", which are mentioned at the starting.
-We also have to change paths for outputs. The path name for weights-file is - "MODEL_OUTPUT_DIR" and "RESULT_OUTPUT_DIR" for storing the results.

DATASET-
-In our dataset, there's column of   "Review_S"(text) and the corresponding images which we are passing as input to the model and the output is in "exp_all" column. This task is done using the function "prepare_dataset" and class "MSE_dataset" in our script file.
-For evaluation code, refer to "get_val_scores" function.

TRAINING-
-Finally we can run the given python file ''exp_model.py". 