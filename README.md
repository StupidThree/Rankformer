# Rankformer: A Graph Transformer for Recommendation based on Ranking Objective

This is the PyTorch implementation for our WWW 2025 paper. 
> Sirui Chen, Shen Han, Jiawei Chen, Binbin Hu, Sheng Zhou, Gang Wang, Yan Feng, Chun Chen, Can Wang. Rankformer: A Graph Transformer for Recommendation based on Ranking Objective
 [arXiv link](https://arxiv.org/abs/2503.16927)

## Environment
- python==3.9.19
- numpy==1.26.4
- pandas==2.2.1
- torch==2.2.2

## Datasets

| Dataset     | #Users  | #Items | #Interactions |
|-------------|---------|--------|---------------|
| Ali-Display | 17,730  | 10,036 | 173,111       |
| Epinions    | 17,893  | 17,659 | 301,378       |
| Amazon-CDs  | 51,266  | 46,463 | 731,734       |
| Yelp2018    | 167,037 | 79,471 | 1,970,721     |

## Training & Evaluation
* Ali-Display
``` bash
# Rankformer
python -u code/main.py --data=Ali-Display \
    --use_gcn \
    --use_rankformer --rankformer_layers=4 --rankformer_tau=0.5 \

# Rankformer-CL
python -u code/main.py --data=Ali-Display \
    --use_cl \
    --use_gcn --gcn_layers=2 --gcn_left=0.5 --gcn_right=0.5 \
    --use_rankformer --rankformer_layers=5 --rankformer_tau=0.1 \
    --learning_rate=1e-3 --loss_batch_size=2048 --valid_interval=1
```
* Epinions
``` bash
# Rankformer
python -u code/main.py --data=Epinions \
    --use_gcn \
    --use_rankformer --rankformer_layers=4 --rankformer_tau=0.4
    
# Rankformer-CL
python -u code/main.py --data=Epinions \
    --use_cl \
    --use_gcn --gcn_layers=2 --gcn_left=0.5 --gcn_right=0.5 \
    --use_rankformer --rankformer_layers=3 --rankformer_tau=0.2 \
    --learning_rate=1e-3 --loss_batch_size=2048 --valid_interval=1
```
* Amazon-CDs
``` bash
# Rankformer
python -u code/main.py --data=Amazon-CDs \
    --use_gcn \
    --use_rankformer --rankformer_layers=2 --rankformer_tau=0.5
    
# Rankformer-CL
python -u code/main.py --data=Amazon-CDs \
    --use_cl \
    --use_gcn --gcn_layers=2 --gcn_left=0.5 --gcn_right=0.5 \
    --use_rankformer --rankformer_layers=4 --rankformer_tau=0.2 \
    --learning_rate=1e-3 --loss_batch_size=2048 --valid_interval=1
```
* Yelp2018
``` bash
# Rankformer
python -u code/main.py --data=Yelp2018 \
    --use_gcn \
    --use_rankformer --rankformer_layers=3 --rankformer_tau=0.4
    
# Rankformer-CL
python -u code/main.py --data=Yelp2018 \
    --use_cl \
    --use_gcn --gcn_layers=2 --gcn_left=0.5 --gcn_right=0.5 \
    --use_rankformer --rankformer_layers=4 --rankformer_tau=0.3 \
    --learning_rate=1e-3 --loss_batch_size=2048 --valid_interval=1
```
  
## Citation
If you find the paper useful in your research, please consider citing:
```
@article{chen2025rankformer,
  title={Rankformer: A Graph Transformer for Recommendation based on Ranking Objective},
  author={Chen, Sirui and Han, Shen and Chen, Jiawei and Hu, Binbin and Zhou, Sheng and Wang, Gang and Feng, Yan and Chen, Chun and Wang, Can},
  journal={arXiv preprint arXiv:2503.16927},
  year={2025}
}
```
