import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, set_color

from ncl import NCL
from trainer import NCLTrainer


def run_single_model(args):
    # configurations initialization
    config = Config(
        model=NCL,
        dataset=args.dataset, 
        config_file_list=args.config_file_list
    )
    init_seed(config['seed'], config['reproducibility'])

    if args.__contains__('alpha'):
        config['alpha'] = args.alpha

    if args.__contains__('ssl_temp'):
        config['ssl_temp'] = args.ssl_temp

    config['loss_type'] = args.loss_type
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = NCL(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = NCLTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m', help='The datasets can be: ml-1m, yelp, amazon-books, gowalla-merged, alibaba.')
    parser.add_argument('--config', type=str, default='', help='External config file name.')
    parser.add_argument('--loss_type', type=int, default=2, help='type 0 : ssl, type 1 : neighbor, type 2 : all')
    parser.add_argument('--alpha', type=float, help='Coefficient alpha between 2 losses of structure loss.')
    parser.add_argument('--ssl_temp', type=float, help='Temperature')

    args, _ = parser.parse_known_args()

    # Config files
    args.config_file_list = [
        'properties/overall.yaml',
        'properties/NCL.yaml'
    ]

    if args.dataset in ['ml-1m', 'yelp', 'amazon-books', 'gowalla-merged', 'alibaba']:
        args.config_file_list.append(f'properties/{args.dataset}.yaml')
    if args.config != '':
        args.config_file_list.append(args.config)

    run_single_model(args)
