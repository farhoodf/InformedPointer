# InformedPointer

## Requirements:

* Pytorch
* Transformers
* Numpy
* Matplotlib
* Pandas
## Dataset
You can download dataset from their own repositories.
* NIPS and SIND can be used without any modification
* ROC and NSF: Code for extracting and dividing into train and test are provide in _prepare_data_

## Train example

python train.py --Path log/nsf/ --Dataset nsf --n_layer_sent 4 --gendim 128 --rnndim 256 --informed_type informed 

## Test example

python test.py --Dataset nips --n_layer_sent 4 --gendim 512 --rnndim 512 --informed_type informed --Path log/nips/1-8model.pt
