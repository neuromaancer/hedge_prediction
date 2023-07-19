"""
@Created Date: Monday January 23rd 2023
@Author: Alafate Abulimiti at alafate.abulimiti@gmail.com
@Company: INRIA
@Lab: CoML/Articulab
@School: PSL/ENS
@Description: Dataset class of 2016 data in tutor's hedge prediction.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from rich import print as rprint
from sentence_transformers import SentenceTransformer
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
)

from rl.reward import HedgeReward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class State:
    turn: str
    cs: dict[str, int]  # cs: conversation strategy
    ts: dict[str, int]  # ts: tutoring strategy
    hedge_label: int  # 0: no hedge, 1: hedge
    nb: dict[str, list]  # nb: nonverbal behavior for tutor and tutee
    da: list = None  # da: dialogue act
    reward_state: dict = None  # information for get reward
    action: int = None  # 0: no hedge, 1: hedge

    @property
    def reward(self):
        reward = HedgeReward(self.reward_state)
        return reward.get_reward()


@dataclass
class Record:
    dyad: int
    session: int
    period: int
    problem_id: int
    state: State


class CollateFn:
    def __init__(self, tensor_list: list[str]):
        self.tensor_list = tensor_list

    def __call__(self, batch):
        labels = [sample["label"].to(device) for sample in batch]
        labels = torch.stack(labels).to(device)
        if not self.tensor_list:
            tensors = [sample["tensor"].to(device) for sample in batch]
            tensors = torch.stack(tensors, dim=-1).to(device)
            tensors = tensors.permute(2, 0, 1)
        else:
            turn_tensors = [sample["turn_tensor"].to(device) for sample in batch]
            turn_tensors = torch.stack(turn_tensors).to(device)
            action_tensors = [sample["action_tensor"].to(device) for sample in batch]
            action_tensors = torch.stack(action_tensors).to(device)
            social_tensors = [sample["social_tensor"].to(device) for sample in batch]
            social_tensors = torch.stack(social_tensors).to(device)
            nb_tensors = [sample["nb_tensor"].to(device) for sample in batch]
            nb_tensors = torch.stack(nb_tensors).to(device)

            interactional_context_tensors = [
                sample["interactional_context_tensor"].to(device) for sample in batch
            ]
            interactional_context_tensors = torch.stack(
                interactional_context_tensors
            ).to(device)

            tensors_dic = {
                "turn": turn_tensors,
                "social": social_tensors,
                "action": action_tensors,
                "nb": nb_tensors,
                "interactional_context": interactional_context_tensors,
            }

            try:
                tensors = [tensors_dic[tensor] for tensor in self.tensor_list]
                tensors = torch.cat(tensors, dim=-1).to(device)
            except KeyError:
                raise KeyError(
                    f"tensor_list should be a list of {list(tensors_dic.keys())}"
                )
        return tensors, labels


class HedgingPredDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sentence_embed_model: SentenceTransformer,
        window_size: int = 4,
        # tensor_input_size=45,
        multi_label: bool = False,
        hierarchical: bool = False,
    ):
        # self.tensor_input_size = tensor_input_size
        self.window_size = window_size
        self.sentence_embed_model = sentence_embed_model
        self.data = df
        # if True, we use subcategories of the hedge label
        self.multi_label = multi_label
        self.hierarchical = hierarchical  # if True, we use hierarchical tensors

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        if idx <= self.window_size:
            y = torch.tensor(0)
            if not self.hierarchical:
                return {
                    "state": {},
                    "record": {},
                    # "tensor": torch.zeros((self.window_size, 437)),
                    "tensor": torch.zeros((self.window_size, 428)),
                    "label": y,
                }
            else:
                return {
                    "state": {},
                    "record": {},
                    "turn_tensor": torch.zeros((self.window_size, 384)),
                    "action_tensor": torch.zeros((self.window_size, 18)),
                    # "action_tensor": torch.zeros((self.window_size, 13)),
                    "social_tensor": torch.zeros(self.window_size, 8),
                    # "social_tensor": torch.zeros(self.window_size, 5),
                    "nb_tensor": torch.zeros(self.window_size, 12),
                    "interactional_context_tensor": torch.zeros(self.window_size, 12),
                    "label": y,
                }
        # If the index is greater than the window size, return the window_size rows before the index
        turns = self.data["turn"].iloc[idx - self.window_size : idx].tolist()
        turns_embeddings = self.sentence_embed_model.encode(
            turns, convert_to_tensor=True, device=device
        )
        # session wise information
        role = self.data["role"].iloc[idx - self.window_size : idx].tolist()
        role_1 = self.data["role_1"][idx - self.window_size : idx].tolist()
        role_2 = self.data["role_2"][idx - self.window_size : idx].tolist()
        session = self.data["session"][idx - self.window_size : idx].tolist()
        session_1 = self.data["session_1"][idx - self.window_size : idx].tolist()
        session_2 = self.data["session_2"][idx - self.window_size : idx].tolist()
        dyad = self.data["dyad"][idx - self.window_size : idx].tolist()
        problem_id = self.data["problem_id"][idx - self.window_size : idx].tolist()
        period = self.data["period"][idx - self.window_size : idx].tolist()
        period_1 = self.data["period_1"][idx - self.window_size : idx].tolist()
        period_2 = self.data["period_2"][idx - self.window_size : idx].tolist()
        # rapport
        rapport = self.data["rapport"][idx - self.window_size : idx].tolist()
        # tutoring strategies
        dq_tutor = self.data["turn_dq_tutor"][idx - self.window_size : idx].tolist()
        dq_tutee = self.data["turn_dq_tutee"][idx - self.window_size : idx].tolist()
        sq_tutor = self.data["turn_sq_tutor"][idx - self.window_size : idx].tolist()
        sq_tutee = self.data["turn_sq_tutee"][idx - self.window_size : idx].tolist()
        kb_tutor = self.data["turn_kb_tutor"][idx - self.window_size : idx].tolist()
        kb_tutee = self.data["turn_kb_tutee"][idx - self.window_size : idx].tolist()
        kt_tutor = self.data["turn_kt_tutor"][idx - self.window_size : idx].tolist()
        kt_tutee = self.data["turn_kt_tutee"][idx - self.window_size : idx].tolist()
        mc_tutor = self.data["turn_mc_tutor"][idx - self.window_size : idx].tolist()
        mc_tutee = self.data["turn_mc_tutee"][idx - self.window_size : idx].tolist()
        # conversational strategies
        pr_tutor = self.data["turn_pr_tutor"][idx - self.window_size : idx].tolist()
        pr_tutee = self.data["turn_pr_tutee"][idx - self.window_size : idx].tolist()
        sd_tutor = self.data["turn_sd_tutor"][idx - self.window_size : idx].tolist()
        sd_tutee = self.data["turn_sd_tutee"][idx - self.window_size : idx].tolist()
        sv_tutor = self.data["turn_sv_tutor"][idx - self.window_size : idx].tolist()
        sv_tutee = self.data["turn_sv_tutee"][idx - self.window_size : idx].tolist()
        hd_tutor = self.data["turn_hd_tutor"][idx - self.window_size : idx].tolist()
        hd_tutee = self.data["turn_hd_tutee"][idx - self.window_size : idx].tolist()
        # nonverbal behaviors
        smile_tutor = self.data["smile_tutor"][idx - self.window_size : idx].tolist()
        smile_tutee = self.data["smile_tutee"][idx - self.window_size : idx].tolist()
        gaze_tutor_0 = self.data["gaze_tutor_0"][idx - self.window_size : idx].tolist()
        gaze_tutor_1 = self.data["gaze_tutor_1"][idx - self.window_size : idx].tolist()
        gaze_tutor_2 = self.data["gaze_tutor_2"][idx - self.window_size : idx].tolist()
        gaze_tutor_3 = self.data["gaze_tutor_3"][idx - self.window_size : idx].tolist()
        gaze_tutee_0 = self.data["gaze_tutee_0"][idx - self.window_size : idx].tolist()
        gaze_tutee_1 = self.data["gaze_tutee_1"][idx - self.window_size : idx].tolist()
        gaze_tutee_2 = self.data["gaze_tutee_2"][idx - self.window_size : idx].tolist()
        gaze_tutee_3 = self.data["gaze_tutee_3"][idx - self.window_size : idx].tolist()
        head_node_tutor = self.data["head_nod_tutor"][
            idx - self.window_size : idx
        ].tolist()
        head_node_tutee = self.data["head_nod_tutee"][
            idx - self.window_size : idx
        ].tolist()

        his_correctness = self.data["correctness"][
            idx - self.window_size : idx
        ].tolist()
        his_problem_id = self.data["problem_id"][idx - self.window_size : idx].tolist()
        his_tutor_learning_gain = self.data["tutor_learning_gain"][
            idx - self.window_size : idx
        ].tolist()
        his_tutee_learning_gain = self.data["tutee_learning_gain"][
            idx - self.window_size : idx
        ].tolist()
        his_tutor_pre_test = self.data["tutor_pre_test"][
            idx - self.window_size : idx
        ].tolist()
        his_tutee_pre_test = self.data["tutee_pre_test"][
            idx - self.window_size : idx
        ].tolist()
        his_tutor_alignment_signals = self.data["tutor_alignment_signals"][
            idx - self.window_size : idx
        ].tolist()
        his_tutee_alignment_signals = self.data["tutee_alignment_signals"][
            idx - self.window_size : idx
        ].tolist()

        # dialog act: dialogue acts are formed as a list, so we need to extend the list to the window size.
        # dialog_act_tutor = self.data["turn_da_tutor"][
        #     idx - self.window_size : idx
        # ].tolist()

        # dialog_act_tutee = self.data["turn_da_tutee"][
        #     idx - self.window_size : idx
        # ].tolist()
        da = self.data["turn_da"][idx - self.window_size : idx].tolist()
        turn_da_0 = self.data["turn_da_0.0"][idx - self.window_size : idx].tolist()
        turn_da_1 = self.data["turn_da_1.0"][idx - self.window_size : idx].tolist()
        turn_da_12 = self.data["turn_da_12.0"][idx - self.window_size : idx].tolist()
        turn_da_17 = self.data["turn_da_17.0"][idx - self.window_size : idx].tolist()
        turn_da_35 = self.data["turn_da_35.0"][idx - self.window_size : idx].tolist()
        turn_da_37 = self.data["turn_da_37.0"][idx - self.window_size : idx].tolist()
        turn_da_39 = self.data["turn_da_39.0"][idx - self.window_size : idx].tolist()
        turn_da_40 = self.data["turn_da_40.0"][idx - self.window_size : idx].tolist()
        # only one label for next turn label prediction.
        # * in the RL model, the turn label is the action.
        label = self.data["next_hedge"][idx]

        if self.multi_label:
            # get the subcategories of the hedge label -> turn_sub_hedge(object)
            raise NotImplementedError("Multi label is not implemented yet.")
        # * 12
        nb = {
            "smile_tutor": smile_tutor,
            "smile_tutee": smile_tutee,
            "gaze_tutor_0": gaze_tutor_0,
            "gaze_tutee_0": gaze_tutee_0,
            "gaze_tutor_1": gaze_tutor_1,
            "gaze_tutee_1": gaze_tutee_1,
            "gaze_tutor_2": gaze_tutor_2,
            "gaze_tutee_2": gaze_tutee_2,
            "gaze_tutor_3": gaze_tutor_3,
            "gaze_tutee_3": gaze_tutee_3,
            "head_node_tutor": head_node_tutor,
            "head_node_tutee": head_node_tutee,
        }

        #! important: the code below make the perf better.
        # if role is Tutor, then the tutor's turn is used.
        # if role == 1:
        #     ts = {
        #         "dq": dq_tutor,
        #         "sq": sq_tutor,
        #         "kb": kb_tutor,
        #         "kt": kt_tutor,
        #         "mc": mc_tutor,
        #     }
        #     cs = {"pr": pr_tutor, "sd": sd_tutor, "sv": sv_tutor, "hd": hd_tutor}
        #     # da = dialog_act_tutor
        # else:
        #     ts = {
        #         "dq": dq_tutee,
        #         "sq": sq_tutee,
        #         "kb": kb_tutee,
        #         "kt": kt_tutee,
        #         "mc": mc_tutee,
        #     }
        #     cs = {"pr": pr_tutee, "sd": sd_tutee, "sv": sv_tutee, "hd": hd_tutee}
        # * 10
        ts = {
            "dq_tutor": dq_tutor,
            "sq_tutor": sq_tutor,
            "kb_tutor": kb_tutor,
            "kt_tutor": kt_tutor,
            "mc_tutor": mc_tutor,
            "dq_tutee": dq_tutee,
            "sq_tutee": sq_tutee,
            "kb_tutee": kb_tutee,
            "kt_tutee": kt_tutee,
            "mc_tutee": mc_tutee,
        }

        # * 8
        cs = {
            "pr_tutor": pr_tutor,
            "sd_tutor": sd_tutor,
            "sv_tutor": sv_tutor,
            "hd_tutor": hd_tutor,
            "pr_tutee": pr_tutee,
            "sd_tutee": sd_tutee,
            "sv_tutee": sv_tutee,
            "hd_tutee": hd_tutee,
        }
        # * 8
        da_dic = {
            "turn_da_0": turn_da_0,
            "turn_da_1": turn_da_1,
            "turn_da_12": turn_da_12,
            "turn_da_17": turn_da_17,
            "turn_da_35": turn_da_35,
            "turn_da_37": turn_da_37,
            "turn_da_39": turn_da_39,
            "turn_da_40": turn_da_40,
        }

        # * add reward
        correctness = self.data["correctness"][idx]
        num_shallow_questions = self.data["tutee_num_shallow_questions"][idx]
        num_deep_questions = self.data["tutee_num_deep_questions"][idx]
        num_alignment_signals = self.data["num_alignment_signals"][idx]

        state_info = {
            "correctness": correctness,
            "learning_gain": 0,
            "num_shallow_questions": num_shallow_questions,
            "num_deep_questions": num_deep_questions,
            "num_alignment_signals": num_alignment_signals,
        }

        state = State(
            turn=turns,
            cs=cs,
            ts=ts,
            da=da_dic,
            nb=nb,
            action=label,
            hedge_label=label,
            reward_state=state_info,
        )

        record = Record(
            dyad=dyad,
            session=session,
            period=period,
            problem_id=problem_id,
            state=state,
        )

        ts_tensor = torch.tensor(
            list(ts.values()), dtype=torch.long, device=device
        ).permute(1, 0)
        # rprint(cs)
        cs_tensor = torch.tensor(
            list(cs.values()), dtype=torch.long, device=device
        ).permute(1, 0)
        rapport_tensor = torch.tensor(
            rapport, dtype=torch.float, device=device
        ).unsqueeze(1)

        nb_tensor = torch.tensor(
            list(nb.values()), dtype=torch.long, device=device
        ).permute(
            1, 0
        )  # 12

        da_tensor = torch.tensor(
            list(da_dic.values()), dtype=torch.long, device=device
        ).permute(1, 0)
        # da_tensor = torch.tensor(da, dtype=torch.long, device=device).unsqueeze(1)
        turn_tensor = turns_embeddings  # 384
        action_tensor = torch.cat((ts_tensor, da_tensor), dim=1)  # 18
        # social_tensor = torch.cat((cs_tensor, rapport_tensor), dim=1)  # 9
        # ablation study 1: without rapport
        social_tensor = cs_tensor  # 8
        # * 14
        interactional_context_tensor = torch.cat(
            (
                torch.tensor(role_1).unsqueeze(1),
                torch.tensor(role_2).unsqueeze(1),
                torch.tensor(session_1).unsqueeze(1),
                torch.tensor(session_2).unsqueeze(1),
                torch.tensor(period_1).unsqueeze(1),
                torch.tensor(period_2).unsqueeze(1),
                torch.tensor(his_problem_id).unsqueeze(1),
                torch.tensor(his_correctness).unsqueeze(1),
      
                torch.tensor(his_tutor_pre_test).unsqueeze(1),
                torch.tensor(his_tutee_pre_test).unsqueeze(1),
                torch.tensor(his_tutor_alignment_signals).unsqueeze(1),
                torch.tensor(his_tutee_alignment_signals).unsqueeze(1),
            ),
            dim=1,
        )
        # the feature tensor dim is: 10 + 8 + 12 + 1 + 8 = 39
        tensor = torch.cat(
            (
                turns_embeddings,
                ts_tensor,
                cs_tensor,
                nb_tensor,
                rapport_tensor,
                da_tensor,
            ),
            dim=1,
        )
        label = torch.tensor(
            label,
            dtype=torch.long,
            device=device,
        )
        if self.hierarchical == True:
            return {
                "state": state,
                "record": record,
                "turn_tensor": turn_tensor,
                "action_tensor": action_tensor,
                "social_tensor": social_tensor,
                "nb_tensor": nb_tensor,
                "interactional_context_tensor": interactional_context_tensor,
                "label": label,
            }
        else:
            return {
                "state": state,
                "record": record,
                "tensor": tensor,
                "label": label,
            }


class HedgingPredDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_folder: str,
        test_data_folder: str,
        batch_size: int,
        num_workers: int,
        sentence_embed_model: SentenceTransformer,
        oversampling_factor: int = 1.0,
        tensor_list: list[str] = None,
    ) -> None:
        super().__init__()
        self.train_data_folder = train_data_folder
        self.test_data_folder = test_data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sentence_embed_model = sentence_embed_model
        self.oversampling_factor = oversampling_factor
        self.tensor_list = tensor_list
        self.prepare_data()

    def _load_dataset(self, path: str, prefix: str = "train") -> ConcatDataset:
        dir_ = Path(path)
        csv_files = list(dir_.glob(prefix + "*.csv"))
        datasets = []
        for csv_file in csv_files:
            data = pd.read_csv(csv_file)
            dataset = HedgingPredDataset(
                df=data,
                sentence_embed_model=self.sentence_embed_model,
                window_size=4,
                hierarchical=self.tensor_list is not None,
            )
            datasets.append(dataset)
        return ConcatDataset(datasets)

    def get_input_size(self):
        input_size = 0
        for tensor in self.tensor_list:
            input_size += self.data[5][f"{tensor}_tensor"].size()[-1]
        return input_size

    def prepare_data(self) -> None:
        self.data = self._load_dataset(self.train_data_folder, prefix="train")
        self.test_set = self._load_dataset(self.test_data_folder, prefix="test")

    def setup(self, stage: Optional[str] = None) -> None:
        # Create Training, Validation and Test Datasets
        #! can't use random split because we need to keep the dyad/session/period/prob_id
        data_length = len(self.data)
        valid_dataset_index = math.ceil(0.85 * data_length)
        self.train_set = Subset(self.data, range(0, valid_dataset_index))
        self.val_set = Subset(self.data, range(valid_dataset_index, data_length))

    def train_dataloader(self) -> DataLoader:
        # oversampling
        targets = [sample["label"] for sample in self.dataset_train]
        targets = torch.tensor(targets, dtype=torch.long, device="cpu")
        class_counts = np.bincount(targets)
        rprint(f"class_counts: {class_counts}")
        weights = 1.0 / torch.Tensor(class_counts)
        weights /= weights.sum()
        rprint(f"weights: {weights}")
        weights *= self.oversampling_factor
        sample_weights = weights[1] * targets + weights[0] * (1 - targets)
        sampler = WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        # create dataloader
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=CollateFn(tensor_list=self.tensor_list),
            sampler=sampler,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=CollateFn(tensor_list=self.tensor_list),
        )

    def update_train_val_datasets(self, train_indices, val_indices):
        self.dataset_train = Subset(self.data, train_indices)
        self.dataset_val = Subset(self.data, val_indices)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=CollateFn(tensor_list=self.tensor_list),
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=CollateFn(tensor_list=self.tensor_list),
        )


if __name__ == "__main__":
    # test `Dataset` Object
    sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    test_df = pd.read_csv(
        "/scratch2/aabulimiti/hedge_prediction/data/testset/all_test_06-03-2023_AA.csv"
    )
    dataset = HedgingPredDataset(test_df, sent_transformer)
    # test `DataModule` Object
    train_data_folder = "/scratch2/aabulimiti/hedge_prediction/data/trainset"
    test_data_folder = "/scratch2/aabulimiti/hedge_prediction/data/testset"
    data_module = HedgingPredDataModule(
        train_data_folder=train_data_folder,
        test_data_folder=test_data_folder,
        batch_size=32,
        num_workers=4,
        sentence_embed_model=sent_transformer,
    )
    data_module.prepare_data()
    data_module.setup()
    rprint(data_module.train_dataloader())
