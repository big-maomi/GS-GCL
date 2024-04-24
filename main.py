import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, set_color
from overrides.local_logger import init_logger

from GS_GCL import GS_GCL
from overrides.trainer import GS_GCLTrainer


def run_single_model(args):
    # Initialize configuration from given arguments
    config = Config(
        model=GS_GCL,
        dataset=args.dataset,
        config_file_list=args.config_file_list
    )
    # Set and log the initial seed for reproducibility
    init_seed(config['seed'], config['reproducibility'])

    # Configure the loss type and training parameters based on arguments
    config['loss_type'] = args.loss_type
    config['train_type'] = []

    if config['loss_type'] != 2:
        config['train_type'].append('loss_type')

    # Conditionally add training configuration parameters if provided
    if args.alpha is not None:
        config['alpha'] = args.alpha
        config['train_type'].append('alpha')

    if args.beta is not None:
        config['beta'] = args.beta
        config['train_type'].append('beta')

    if args.ssl_temp is not None:
        config['ssl_temp'] = args.ssl_temp
        config['train_type'].append('ssl_temp')

    if args.epochs is not None:
        config['epochs'] = args.epochs
        config['train_type'].append('epochs')

    if args.proto_reg is not None:
        config['proto_reg'] = float(args.proto_reg)
        config['train_type'].append('proto_reg')

    # Initialize logging
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # Load and log dataset
    dataset = create_dataset(config)
    logger.info(dataset)

    # Prepare data for training, validation, and testing
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Initialize model and log configuration
    model = GS_GCL(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # Initialize trainer
    trainer = GS_GCLTrainer(config, model)

    # Execute training process and capture the best validation score
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # Evaluate the model on test data and log the results
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m', help='The datasets can be: ml-1m, yelp, amazon-books, gowalla-merged, alibaba.')
    parser.add_argument('--config', type=str, default='', help='External config file name.')
    parser.add_argument('--loss_type', type=int, default=2, help='type 0 : ssl, type 1 : neighbor, type 2 : all')
    parser.add_argument('--alpha', type=float, help='Coefficient alpha between 2 losses of structure loss.')
    parser.add_argument('--beta', type=float, help='Coefficient beta between 2 losses of nearest loss.')
    parser.add_argument('--ssl_temp', type=float, help='Temperature')
    parser.add_argument('--epochs', type=int, help='Epochs')
    parser.add_argument('--proto_reg', type=str, help='proto_reg')

    args, _ = parser.parse_known_args()

    # Config files
    args.config_file_list = [
        'properties/overall.yaml',
        'properties/GS_GCL.yaml'
    ]

    if args.dataset in ['ml-1m', 'yelp', 'amazon-books', 'gowalla-merged', 'alibaba']:
        args.config_file_list.append(f'properties/{args.dataset}.yaml')
    if args.config != '':
        args.config_file_list.append(args.config)

    run_single_model(args)
