import copy
import random
import gc

import numpy as np
import torch
import logging
import argparse

from participants.clients.BenignClient import BenignClient
from participants.clients.MaliciousClient import MaliciousClient
from participants.clients.MirageClient import MirageClient
from participants.clients.AFDMClient import AFDMClient

from participants.servers.No_defense_Server import No_defense_Server

from utils.utils import args_update
from datasets.MSP_dataloader import MSPDataloader

logger = logging.getLogger("logger")


# ==========================================================
# RANDOM SEED
# ==========================================================

def set_random_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    yaml_file = "yamls/AFDM/AFDM_nodefense.yaml"

    parser.add_argument(
        "--params",
        default=f"{yaml_file}",
        type=str
    )

    parser.add_argument(
        "--no_of_adversaries",
        default=2,
        type=int
    )

    parser.add_argument(
        "--poison_type",
        default="continue_poison",
        type=str
    )

    parser.add_argument(
        "--attach",
        default="",
        type=str
    )

    parser.add_argument(
        "--gpu_id",
        default="0",
        type=str
    )

    parser.add_argument(
        "--model_type",
        default="ResNet18",
        type=str
    )

    parser.add_argument(
        "--dataset",
        default="GTSRB",
        type=str
    )

    args = parser.parse_args()

    params_loaded = args_update(args)

    # ==========================================================
    # DATASET SETTINGS
    # ==========================================================

    if params_loaded["dataset"].upper() == "CIFAR10":

        params_loaded["class_num"] = 10
        params_loaded["img_size"] = 32
        params_loaded["input_channel"] = 3

    elif params_loaded["dataset"].upper() == "CIFAR100":

        params_loaded["class_num"] = 100
        params_loaded["img_size"] = 32
        params_loaded["input_channel"] = 3

    elif params_loaded["dataset"].upper() == "EMNIST":

        params_loaded["class_num"] = 10
        params_loaded["img_size"] = 28
        params_loaded["input_channel"] = 1

    elif params_loaded["dataset"].upper() == "GTSRB":

        params_loaded["class_num"] = 43

        params_loaded["poison_train_batch_size"] = 32
        params_loaded["train_batch_size"] = 32
        params_loaded["poisoned_len"] = 4

        # AFDM important
        params_loaded["img_size"] = 32
        params_loaded["input_channel"] = 3

    else:
        raise NotImplementedError

    logger.info(
        f'Resumed Model: {params_loaded["resumed_model"]}'
    )

    logger.info(f"Params: {params_loaded}")

    # ==========================================================
    # RANDOM SEED
    # ==========================================================

    set_random_seed(params_loaded["seed"])

    # ==========================================================
    # DATALOADER
    # ==========================================================

    dataloader = MSPDataloader(params_loaded)

    # ==========================================================
    # SERVER
    # ==========================================================

    if params_loaded["defense_method"].lower() == "nodefense":

        server = No_defense_Server(
            params=params_loaded,
            dataloader=dataloader
        )

    else:
        raise NotImplementedError

    # ==========================================================
    # BENIGN CLIENT
    # ==========================================================

    benign_client = BenignClient(
        params_loaded,
        dataloader.train_dataloader,
        dataloader.test_dataloader
    )

    # ==========================================================
    # MALICIOUS CLIENT
    # ==========================================================

    if params_loaded["malicious_train_algo"] == "Mirage":

        malicious_client = MirageClient(
            params_loaded,
            dataloader.train_dataloader,
            dataloader.test_dataloader
        )

    elif params_loaded["malicious_train_algo"] == "AFDM":

        malicious_client = AFDMClient(
            params_loaded,
            dataloader.train_dataloader,
            dataloader.test_dataloader
        )

    else:

        malicious_client = MaliciousClient(
            params_loaded,
            dataloader.train_dataloader,
            dataloader.test_dataloader
        )

    # ==========================================================
    # TRAIN LOOP
    # ==========================================================

    for iteration in range(
            server.params["start_iteration"],
            server.params["end_iteration"]
    ):

        logger.info(
            f"====================== Current Round: {iteration} ======================"
        )

        # ==========================================================
        # PRE-PROCESS
        # ==========================================================

        server.pre_process(
            test_data=server.test_dataloader,
            iteration=iteration
        )

        # ==========================================================
        # CLIENT TRAINING + UPLOAD
        # ==========================================================

        (
            weight_accumulator,
            weight_accumulator_by_client,
            aggregated_model_id
        ) = server.broadcast_upload(
            iteration=iteration,
            benign_client=benign_client,
            malicious_client=malicious_client
        )

        # ==========================================================
        # SERVER AGGREGATION
        # ==========================================================

        server.aggregation(
            weight_accumulator=weight_accumulator,
            aggregated_model_id=aggregated_model_id
        )

        logger.info(
            f"Aggregated Model ID: {aggregated_model_id}"
        )

        # ==========================================================
        # GLOBAL MODEL TEST
        # ==========================================================

        server.test_global_model(
            iteration=iteration,
            malicious_clients=malicious_client
        )

        # ==========================================================
        # SAVE MODEL
        # ==========================================================

        server.save_model(
            iteration,
            malicious_client.trigger_set,
            malicious_client.mask_set
        )

        # ==========================================================
        # MEMORY CLEANUP
        # ==========================================================

        try:
            del weight_accumulator
            del weight_accumulator_by_client
            del aggregated_model_id

        except:
            pass

        gc.collect()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    logger.info("Training Finished Successfully")