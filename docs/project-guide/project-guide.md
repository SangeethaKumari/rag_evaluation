# Getting started with the labs

You will find the lab notes and other related documentation that will help you with your project.

## Config file paths
Update below entries of the `config.yaml` file of the project's root folder to point to suitable paths in your local:

```yaml
paths:
  books_dir: /path/to/text_chunks # input dir for text chunks
  results: /path/to/model_output_dir # for model output of text and image classification
  data: /path/to/trees_data # input dir for tree images

mnist-classification:
  data: /path/to/mnist_data # data dir for mnist images
```

## Data required for labs
The `attention_basics.ipynb` will download the `MNIST` data into the folder specified above for the first time and then subsequent runs will pick the data from here. 

The text transformer python source code for training and the jupyter notebook for evaluating/visualization use the `books_dir` and `results` path variables from the `config.yaml`, while the vision transformer uses the `data` for the tree images and the same `results` as the output dir for the trained model.

The required data for these can be downloaded from the `llm bootcamp datasets` folder in google drive:
1. For text chunks: `chunks.tar.gz`
2. For tree images: `trees_dataset.tar.gz`