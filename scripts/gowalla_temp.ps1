conda activate RecBole

python main.py --dataset gowalla-merged --ssl_temp 0.05
python main.py --dataset gowalla-merged --ssl_temp 0.075
python main.py --dataset gowalla-merged --ssl_temp 0.100
python main.py --dataset gowalla-merged --ssl_temp 0.125
python main.py --dataset gowalla-merged --ssl_temp 0.150
