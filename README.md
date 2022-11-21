# kengic

## Overview
An image captioning model based on N-Gram graphs which does not require any end-to-end training on paired image captions. 
Starting with a set of image keywords considered as nodes, the generator is designed to form a directed graph by connecting these nodes
through overlapping n-grams as found in a given text corpus. The model then infers the caption by maximising the most probable n-gram sequences from the constructed graph.

## Publication
KENGIC: KEyword-driven and N-gram Graph based Image Captioning <br/>
[Brandon Birmingham](https://brandbir.github.io) and [Adrian Muscat](https://www.um.edu.mt/profile/adrianmuscat) <br/>
[DICTA 2022](http://dicta2022.dictaconference.org/)

## Dependencies

- ``$ pip installl -U scikit-learn``
- ``$ conda install pandas``
- ``$ pip install --user -U nltk``
- Download [COCO dataset (Karpathy's split)](https://drive.google.com/file/d/1yMGYPcRepSakuMnpuBYcz2Vug_wKZAKY/view?usp=share_link) into `/kengic/src/`

## Folder structure
- `images`: set of query images
- `src`: source files
    - `dataset_coco_json`: coco dataset
    - `stopwords.json`: set of stopwords
    - `tools.py`: generic functions
    - `nlp.py`: nlp related functions
    - `src.py`: main kengic code
    - `runner.py` kengic runner file
- `out`: output folder
    - `KGCap.log`: kengic logs file
    - `out_i.csv`: output file of process i
    - `out.csv`: combined output of all processes
    - `combined_imgs_ids.txt`: list of img ids which were not captioned

## Arguments
Check command line arguments by `$ python3 runner.py -h:`

```
usage: runner.py [-h] [--n N] [--p P] [--h H] [--nlogP NLOGP] [--optimiser OPTIMISER] [--i I]`
                 `[--coco-splits-path COCO_SPLITS_PATH] [--input-file INPUT_FILE] [--out-dir OUT_DIR] [--num-imgs NUM_IMGS]
                 [--column-name COLUMN_NAME] [--keywords-type KEYWORDS_TYPE] [--keywords-split KEYWORDS_SPLIT]
                 [--start-end-tokens START_END_TOKENS] [--top-n-captions TOP_N_CAPTIONS] [--num-processes NUM_PROCESSES]
                 [--stopwords-path STOPWORDS_PATH]

KGCap

optional arguments:
  -h, --help            show this help message and exit
  --n N                 ngrams [3,4,5] (default: 3)
  --p P                 parents (default: 5)
  --h H                 hops (default: 1)
  --nlogP NLOGP         nlogP [3,4,5] (default: 3)
  --optimiser OPTIMISER
                        optimiser [1,2,3,4] (default: 1)
  --i I                 traversal iterations (default: 150)
  --coco-splits-path COCO_SPLITS_PATH
                        Karpathy COCO splits path (default:dataset_coco.json)
  --input-file INPUT_FILE
                        input csv file
  --out-dir OUT_DIR     output directory
  --num-imgs NUM_IMGS   number of images to caption
  --column-name COLUMN_NAME
                        input csv file column name for keywords
  --keywords-type KEYWORDS_TYPE
                        Type of keywords to use: {"all", "hypernyms", "hyponyms" (default: "all")
  --keywords-split KEYWORDS_SPLIT
                        whether multi-word keywords are split or not (default: False)
  --start-end-tokens START_END_TOKENS
                        whether to use start/end tokens (default False)
  --top-n-captions TOP_N_CAPTIONS
                        number of captions to be considered during caption generation (default: 5
  --num-processes NUM_PROCESSES
                        number of processes (default: 1)
  --stopwords-path STOPWORDS_PATH
                        Stopwords file path (default stopwords.csv)
```

## Execution
 - `cd /kengic/src`
 - `$ python3 runner.py --input-file ../keywords_predicted.csv --column-name pred_classes --out-dir out --num-imgs 4 --num-processes 4`:

 ```
 ===========================================================================
Running KGCap (n: 3 parents: 5 hops: 1 nlogP: 3 optimiser: 4 max_iters: 150)
---------------------------------------------------------------------------
input_csv_file_path: ../keywords_predicted.csv
num_of_images: 4
keywords_column: pred_classes
keywords_type: all
keywords_split: False
include_start_end_tokens: False
top_n_captions: 5
out_folder: out
num_of_processes: 4
===========================================================================
------------------------------------------------
Reading karpathy splits...
------------------------------------------------
Loading splits...
------------------------------------------------
Loading n-grams...
1 of 113287
10001 of 113287
20001 of 113287
30001 of 113287
40001 of 113287
50001 of 113287
60001 of 113287
70001 of 113287
80001 of 113287
90001 of 113287
100001 of 113287
110001 of 113287
------------------------------------------------
Loading n-grams into DataFrames...
------------------------------------------------
Loading stopwords...
=======================================================
Loading input file...
Total number of images to caption:4
Number of processes:4
Starting processes..
Waiting for processes to finish...
------------------------------------------------------------------------------------------
pid:0 1/1 - img_id:20078 pred_classes:['snow', 'street', 'ski', 'woman', 'walk', 'skiing', 'people']
------------------------------------------------------------------------------------------
pid:0 - Filtering keywords...
pid:0 - Splitting composite keywords if not found in training data
pid:0 - Updated keywords:['snow', 'street', 'ski', 'woman', 'walk', 'skiing', 'people']
pid:0 - Filtered keywords: ['snow', 'street', 'ski', 'woman', 'walk', 'skiing', 'people']
pid:0 - Removing stopwords
pid:0 - Generating graph for keywords:['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] (n:3 neighbourhood:5 hops:1)
pid:0 - Graph generated in: 0.02 minutes
pid:0 - Post-processing graph (size:33) keywords:['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] min conns:5
------------------------------------------------------------------------------------------
pid:1 1/1 - img_id:13910 pred_classes:['dog', 'skateboard', 'leash']
------------------------------------------------------------------------------------------
pid:1 - Filtering keywords...
pid:1 - Splitting composite keywords if not found in training data
pid:1 - Updated keywords:['dog', 'skateboard', 'leash']
pid:1 - Filtered keywords: ['dog', 'skateboard', 'leash']
pid:1 - Removing stopwords
pid:1 - Generating graph for keywords:['dog', 'skateboard', 'leash'] (n:3 neighbourhood:5 hops:1)
pid:1 - Graph generated in: 0.01 minutes
pid:1 - Post-processing graph (size:15) keywords:['dog', 'skateboard', 'leash'] min conns:5
------------------------------------------------------------------------------------------
pid:2 1/1 - img_id:21711 pred_classes:['bus', 'double', 'decker', 'green', 'park', 'three']
------------------------------------------------------------------------------------------
pid:2 - Filtering keywords...
pid:2 - Splitting composite keywords if not found in training data
pid:2 - Updated keywords:['bus', 'double', 'decker', 'green', 'park', 'three']
pid:2 - Filtered keywords: ['bus', 'double', 'decker', 'green', 'park', 'three']
pid:2 - Removing stopwords
pid:2 - Generating graph for keywords:['decker', 'bus', 'green', 'park', 'double', 'three'] (n:3 neighbourhood:5 hops:1)
pid:2 - Graph generated in: 0.01 minutes
pid:2 - Post-processing graph (size:28) keywords:['decker', 'bus', 'green', 'park', 'double', 'three'] min conns:5
------------------------------------------------------------------------------------------
pid:3 1/1 - img_id:36003 pred_classes:['mirror', 'bus', 'school', 'view']
------------------------------------------------------------------------------------------
pid:3 - Filtering keywords...
pid:3 - Splitting composite keywords if not found in training data
pid:3 - Updated keywords:['mirror', 'bus', 'school', 'view']
pid:3 - Filtered keywords: ['mirror', 'bus', 'school', 'view']
pid:3 - Removing stopwords
pid:3 - Generating graph for keywords:['view', 'bus', 'school', 'mirror'] (n:3 neighbourhood:5 hops:1)
pid:3 - Graph generated in: 0.01 minutes
pid:3 - Post-processing graph (size:18) keywords:['view', 'bus', 'school', 'mirror'] min conns:5

.
.
.

------------------------------------------------------------------------------------------

pid:0 22/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:4 captions:1
pid:0 23/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:7 captions:1
pid:0 24/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:8 captions:1
pid:0 25/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:4 captions:2
pid:0 26/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:5 captions:2
pid:0 27/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:6 captions:2
pid:0 28/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:5 captions:2
pid:0 29/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:4 captions:3
pid:0 30/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:4 captions:3
pid:0 31/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:3 captions:4
pid:0 32/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:4 captions:4
pid:0 33/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:2 captions:4
pid:0 34/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:5 captions:4
pid:0 35/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:7 captions:4
pid:0 36/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:4 captions:5
pid:0 37/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:3 captions:6
pid:0 38/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:3 captions:6
pid:0 39/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:2 captions:7
pid:0 40/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:4 captions:7
pid:0 41/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:3 captions:8
pid:0 42/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:2 captions:8
pid:0 43/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:2 captions:8
pid:0 44/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:1 captions:8
pid:0 45/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:1 captions:8
pid:0 46/150 ['people', 'skiing', 'ski', 'woman', 'snow', 'walk', 'street'] paths:0 captions:9
pid:0 - Computing log prob for captions...
pid:0 - Graph traversed in: 4.71 minutes
pid:0 - Generating captions metrics DataFrame...
pid:0 - Total duration: 6.91 minutes
[process_detections] pid:0 - Writing captions to out/out_0.csv
------------------------------------------------------------------------------------------
Process:0 duration: 6.91 minutes
------------------------------------------------------------------------------------------

Processes finished...
Combining output...
Combining out/out_0.csv
Combining out/out_1.csv
Combining out/out_2.csv
Combining out/out_3.csv
Total duration: 7.12 minutes
```

## Output
 - logs: `/kengic/src/out/KGCap.log`
 - captions: `out.csv` (in column named caption)


## Results

|  | |  | |
| ----------- | ----------- | ----------- | ----------- |
| <img src="images/COCO_val2014_000000002890.jpg" height="200" /> | <img src="images/COCO_val2014_000000438915.jpg" height="200" /> | <img src="images/COCO_val2014_000000090208.jpg" height="200" /> | <img src="images/COCO_val2014_000000291538.jpg" height="200" /> |
| People walk down a snow covered street <br/> and a  woman skiing in the snow on a ski. | Dog wearing a leash on a skateboard. | Three double decker bus. | View of a mirror in a school bus.|
||||