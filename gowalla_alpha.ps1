conda activate RecBole

python main.py --dataset gowalla-merged --alpha 0.1
python main.py --dataset gowalla-merged --alpha 0.3
python main.py --dataset gowalla-merged --alpha 0.5
python main.py --dataset gowalla-merged --alpha 0.8
python main.py --dataset gowalla-merged --alpha 1.0
python main.py --dataset gowalla-merged --alpha 1.2
python main.py --dataset gowalla-merged --alpha 1.5
python main.py --dataset gowalla-merged --alpha 1.8
python main.py --dataset gowalla-merged --alpha 2.0

