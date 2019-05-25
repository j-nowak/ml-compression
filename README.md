# Quantization in Compressive Autoencoders
In this repository different techniques of quantization in compressive neural networks are evaluated.

## Prerequisite
Install Conda
```shell
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Create a new environment from spec
`conda create --name networks-do-networks --file conf/spec-file.txt`

Activate the env
`source activate networks-do-networks`

Export required environment variables
```shell
export LD_LIBRARY_PATH=<PATH_TO_YOUR_CONDA>/conda/miniconda3/envs/tf/lib/ 
export CUDA_VISIBLE_DEVICES=0
```

## Running model
There are two starting scripts depending on a chosen dataset:
* `celeba_train.py` for training and testing on CelebA dataset,
* `clic_train` for training on images bigger than `64x64` pixels. In my settings we used combined `CLIC` nd `DIV2K` datasests for training and `Kodak` dataset for testing.

General entry script template to start training is:
```
python3 <STARTING_SCRIPT> --dataset <TRAINING_DATASET> --test_dataset <TESTING_DATASET> --img_x <IMG_WIDTH> --img_y <IMG_HEIGHT> --test_per_iterations <STEPS_BETWEEN_EVALS> --steps <NUM_OF_STEPS> --batch_size <BATCH_SIZE> --model_name <MODEL_NAME> --model_type <MODEL_TYPE> --quant_size <QUANT_SIZE>
```

There are also other tunable parameters. For specifics please see the source code or contact us.