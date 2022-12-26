# Neural_gender_classification
A Pytorch Neural Network that classifies face images by gender (male, female)

SETUP:
At first, I highly recomend using PyCharm.

1. Download Python 3.8, open cmd, and type 'pip install poetry'.
2. Clone the repository.
3. After cloning the repository, create a new project in PyCharm and select the repo folder.
4. Click '<no interpreter>' and add new interpreter at right down, after that, click add local interpreter.
5. Click Poetry envitoment, select your python base interpreter and the poetry (you installed at 1.).
6. On PyCharm project cmd, type 'poetry update and run', and all the packages should be installed.

TRAIN:
You can create your own dataset and than train the model, or run with the dataset provided.
To do so, you just need to enter the scripts folder "gender reco", and run "python train.py dir/to/train dir/to/test",
you can change either epochs values on the script.

PREDICT:
After training the model, you can make some predicts with your own photos, you just need to enter the scripts
path, and run "python predict.py dir/to/model image/path"

NOTES:
Please, if you notice any problem, just let me know.
