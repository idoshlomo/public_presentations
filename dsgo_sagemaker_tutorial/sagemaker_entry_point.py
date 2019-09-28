import argparse
import pandas as pd
import os

import logging
import json
import custom_code  # importing whatever custom code you have
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# THIS HANDLES TRAINING (DEFAULT SCRIPT INVOKE)
if __name__ == '__main__':

    # parse environment variables
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    args = parser.parse_args()

    # read training data from train directory
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(input_files) == 0:
        raise ValueError('There are no files in {}'.format(args.train, "train"))

    logger.info('reading input files: {}'.format(input_files))
    raw_data = [pd.read_csv(file) for file in input_files]
    train_data = pd.concat(raw_data)

    # fit model to data
    name_comparison_model = custom_code.fit_model(train_data)

    # save model as file
    custom_code.save_model(name_comparison_model, args.model_dir)


# THIS LOADS A TRAINED MODEL
def model_fn(model_dir):
    mdl = custom_code.load_model(model_dir)
    return mdl


# THIS HANDLES INPUT SENT TO THE MODEL
def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        in_data = json.loads(request_body)
        return in_data
    else:
        raise ValueError("bad input")


# THIS APPLIES MODEL TO INPUT AND RETURNS PREDICTION
def predict_fn(input_data, model):
    mdl_output = custom_code.use_model(input_data, model)
    return mdl_output
