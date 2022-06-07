# Image Matching

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-OpenCV](https://img.shields.io/badge/Made%20with-OpenCV-green)](https://opencv.org/)


## What's This Project About?

You may have hundreds or even thousands of images when you come back from several
trips. I bet they are pretty much un-organized (i.e., you don’t have to sort, categorize, and
annotate them). So, you end up with a large set of images sitting in a folder. Each time
when you want to find some photos to share with your friends, you may have to browse and
search them one by one from the very beginning. This is so frustrating. And we need a
smart way for this purpose. This is indeed an image retrieval system, in which you can do
googling on images (over your own image databases).

This is a simple application of ORB descriptors. A database is created for image
path and features storing.

### Repo Structure

```
.
├── LICENSE
├── Pipfile
├── Pipfile.lock
├── README.md
├── imageMeta.db
├── img/          
|   ├── images/       # data directory that stores the photo from flickr dataset
|   └── captions.csv  # caption of the images
├── scripts/          # scripts for automation
├── results/          # results of the query
└── src/              # Srouce code
```

### Dataset

[Flickr 8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) is used to simulate
the daily life photos.


## Dependencies

The dependencies are managed by `Pipfile`, and an exported `requirements.txt` is
available. To install the dependencies, use your favorite package management system
to install via `requirements.txt` or use pipenv. Run `pipenv install` to install.

## How to Run?

1. If your are using other package management system, activate your virtual environment,
and:
  - run `python src/image_check.py` to setup the database.
  - run
  `python src/matching.py -f "Relative path in terms of image_matching directory to the query image." -m "ORB or KAZE"`
  to match similar photos.

2. If you are using pipenv:
  - run `pipenv run check` to setup the database
  - run
  `pipenv run matching "Relative path in terms of image_matching directory to the query image." "ORB or KAZE"`
  to match similar photos

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.
