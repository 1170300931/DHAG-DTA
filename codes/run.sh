python codes/infer.py --dataset davis --split_mode S1 --cuda_id 0 --ratio 0.05 --skip True
python codes/infer.py --dataset kiba --split_mode S1 --cuda_id 0 --ratio 0.0115 --skip True

python codes/infer.py --dataset davis --split_mode S2 --cuda_id 0 --ratio 0.05 --skip True --sim_d 5 --sim_t 10
python codes/infer.py --dataset kiba --split_mode S2 --cuda_id 0 --ratio 0.0115 --skip True --sim_d 6 --sim_t 10

python codes/infer.py --dataset davis --split_mode S3 --cuda_id 0 --ratio 0.05 --skip True --sim_d 5 --sim_t 10
python codes/infer.py --dataset kiba --split_mode S3 --cuda_id 0 --ratio 0.0115 --skip True --sim_d 6 --sim_t 3

python codes/infer.py --dataset davis --split_mode S4 --cuda_id 0 --ratio 0.05 --skip True --sim_d 5 --sim_t 10
python codes/infer.py --dataset kiba --split_mode S4 --cuda_id 0 --ratio 0.0115 --skip True --sim_d 6 --sim_t 3