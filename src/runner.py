from src import KGCap
import argparse

coco_splits_path = ''

parser = argparse.ArgumentParser(description='KGCap')
parser.add_argument('--n',                default=3,  type=int, help='ngrams [3,4,5] (default: 3)')
parser.add_argument('--p',                default=5,  type=int, help='parents (default: 5)')
parser.add_argument('--h',                default=1,  type=int, help='hops (default: 1)')
parser.add_argument('--nlogP',            default=3,  type=int, help='nlogP [3,4,5] (default: 3)')
parser.add_argument('--optimiser',        default=4,  type=int, help='optimiser [1,2,3,4] (default: 1)')
parser.add_argument('--i',                default=150, type=int, help='traversal iterations (default: 150)')
parser.add_argument('--coco-splits-path', default='dataset_coco.json', type=str, help='Karpathy COCO splits path (default:' + coco_splits_path + ')')
parser.add_argument('--input-file',       default='../detections/hk_f_intersection_0.csv',  type=str, help='input csv file')
parser.add_argument('--out-dir',          default='out',  type=str, help='output directory')
parser.add_argument('--num-imgs',         default=5000,  type=int, help='number of images to caption')
parser.add_argument('--column-name',      default='hf-3+4+5',  type=str, help='input csv file column name for keywords')
parser.add_argument('--keywords-type',    default='all',  type=str, help='Type of keywords to use: {"all", "hypernyms", "hyponyms" (default: "all")')
parser.add_argument('--keywords-split',   default=False,  type=bool, help='whether multi-word keywords are split or not (default: False)')
parser.add_argument('--start-end-tokens', default=False,  type=bool, help='whether to use start/end tokens (default False)')
parser.add_argument('--top-n-captions',   default=5,  type=int, help='number of captions to be considered during caption generation (default: 5')
parser.add_argument('--num-processes',    default=2,  type=int, help='number of processes (default: 1)')
parser.add_argument('--stopwords-path',   default='stopwords.csv',  type=str, help='Stopwords file path (default stopwords.csv)')


args = parser.parse_args()

def main():
    n = args.n
    parents = args.p
    hops = args.h
    nlogP = args.nlogP
    optimiser = args.optimiser
    max_iters = args.i
    configs = [n, parents, hops, nlogP, optimiser, max_iters]
    
    captioner = KGCap(configs,
                      args.coco_splits_path,
                      args.stopwords_path,
                      out_folder=args.out_dir,
                      input_csv_file_path=args.input_file, 
                      num_imgs=args.num_imgs,
                      keywords_column=args.column_name,
                      keywords_type=args.keywords_type,
                      keywords_split=args.keywords_split,
                      include_start_end_tokens=args.start_end_tokens,
                      top_n_captions=args.top_n_captions,
                      num_processes=args.num_processes)
    
    captioner.generate_captions()

if __name__ == '__main__':
    main()
