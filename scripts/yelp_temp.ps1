conda activate RecBole

python main.py --dataset yelp --ssl_temp 0.05
python main.py --dataset yelp --ssl_temp 0.075
python main.py --dataset yelp --ssl_temp 0.100
python main.py --dataset yelp --ssl_temp 0.125
python main.py --dataset yelp --ssl_temp 0.150