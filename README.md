## Installation
```
conda create -n fewshot python=3.6
source activate fewshot
pip install -r requirements.txt
```

## Replicating Results
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py \
--model="gpt2-xl" \
--dataset="toxicbias, trec, cb, agnews, dbpedia" \
--num_seeds=5 \
--all_shots="0, 1, 4, 8" \
--subsample_test_set=300 \
--approx
```
## Citation
```
@article{Zhao2021Calibrate,	
  Author = {Tony Z. Zhao and Eric Wallace and Shi Feng and Dan Klein and Sameer Singh},	
  Journal={arXiv preprint arXiv:2102.09690},	
  Year = {2021},	
  Title = {Calibrate Before Use: Improving Few-shot Performance of Language Models}	
}    	
```
