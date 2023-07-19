import logging
from datetime import datetime

import click
from sentence_transformers import SentenceTransformer

from dataset import HedgingPredDataModule
from preds.dummy import DummyClassifier

train_data_folder = "/scratch2/aabulimiti/hedge_prediction/data/trainset"
test_data_folder = "/scratch2/aabulimiti/hedge_prediction/data/testset"
sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")


@click.command()
@click.option(
    "-l",
    "--log_file",
    default="../logs/dummy/dummy.log",
    type=str,
    help="Name of the log file.",
)
def train_dummy(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Dummy classifier")
    logging.info(f"today: {datetime.now()}")

    data_module = HedgingPredDataModule(
        train_data_folder=train_data_folder,
        test_data_folder=test_data_folder,
        batch_size=6,
        num_workers=0,
        sentence_embed_model=sent_transformer,
    )

    data_module.setup()
    data_module.prepare_data()

    clf = DummyClassifier(data_module.train_set, data_module.test_set)
    clf.fit()
    result = clf.predict()
    logging.info(result)
if __name__ == "__main__":
    train_dummy()
