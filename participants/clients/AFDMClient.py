import copy
import torch
import logging
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm

import torch_dct as dct

from participants.clients.BasicClient import BasicClient
from utils.utils import poisoned_batch_injection

logger = logging.getLogger("logger")


class AFDMClient(BasicClient):
    """
    Adaptive Frequency-Domain Mirage (AFDM)

    Main Novelty:
    - Frequency-domain trigger optimization
    - Adaptive spectral modulation
    - Dynamic trigger evolution
    - Spectral manifold alignment
    """

    def __init__(self, params, train_dataloader, test_dataloader):

        super(AFDMClient, self).__init__(
            params,
            train_dataloader,
            test_dataloader
        )

        self.init_trigger_mask()
        self.init_frequency_trigger()

    # ==========================================================
    # INITIALIZE FREQUENCY TRIGGERS
    # ==========================================================

    def init_frequency_trigger(self):

        self.freq_trigger_set = {}
        self.spectral_memory = {}

        img_size = self.params["img_size"]
        channels = self.params["input_channel"]

        for client_id in range(self.params["no_of_adversaries"]):

            freq_trigger = torch.randn(
                channels,
                img_size,
                img_size,
                device=self.params["run_device"]
            ) * 0.01

            self.freq_trigger_set[client_id] = freq_trigger

            self.spectral_memory[client_id] = torch.zeros_like(
                freq_trigger
            )

    # ==========================================================
    # DCT / IDCT
    # ==========================================================

    def dct2d(self, x):
        return dct.dct_2d(x)

    def idct2d(self, x):
        return dct.idct_2d(x)

    # ==========================================================
    # SPECTRAL ATTENTION
    # ==========================================================

    def compute_spectral_attention(
            self,
            feature_statistics,
            client_id):

        attention = torch.sigmoid(feature_statistics)

        attention = attention / (
                attention.mean() + 1e-8
        )

        self.spectral_memory[client_id] = (
                0.9 * self.spectral_memory[client_id]
                +
                0.1 * attention.detach()
        )

        attention = (
                attention
                +
                self.spectral_memory[client_id]
        )

        return attention

    # ==========================================================
    # FREQUENCY SMOOTHNESS LOSS
    # ==========================================================

    def frequency_smoothness_loss(self, freq_trigger):

        diff_h = (
                freq_trigger[:, :, 1:]
                -
                freq_trigger[:, :, :-1]
        )

        diff_w = (
                freq_trigger[:, 1:, :]
                -
                freq_trigger[:, :-1, :]
        )

        return (
                diff_h.abs().mean()
                +
                diff_w.abs().mean()
        )

    # ==========================================================
    # SPECTRAL ALIGNMENT LOSS
    # ==========================================================

    def spectral_alignment_loss(
            self,
            clean_freq,
            poison_freq):

        return F.mse_loss(
            clean_freq,
            poison_freq
        )

    # ==========================================================
    # ADAPTIVE FREQUENCY MODULATION
    # ==========================================================

    def adaptive_frequency_modulation(
            self,
            freq_trigger,
            client_id,
            feature_statistics,
            freq_mask):

        attention = self.compute_spectral_attention(
            feature_statistics,
            client_id
        )

        adaptive_trigger = freq_trigger * attention

        adaptive_trigger = adaptive_trigger * freq_mask

        return adaptive_trigger

    # ==========================================================
    # FEATURE EXTRACTOR
    # ==========================================================

    def get_feature_extractor(self, model):

        feature_model = copy.deepcopy(model)

        if "resnet" in self.params["model_type"].lower():
            feature_model.linear = nn.Identity()

        feature_model.eval()

        return feature_model

    # ==========================================================
    # GENERATE DISCRIMINATOR DATALOADER
    # ==========================================================

    def generate_discriminator_dataloader(
            self,
            model,
            train_loader,
            trigger_,
            mask_,
            client_id):

        class_num = self.params["class_num"]

        samples_per_class = {
            i: torch.tensor([], device=self.params["run_device"])
            for i in range(class_num)
        }

        criterion = nn.CrossEntropyLoss(
            reduction='none'
        ).to(self.params["run_device"])

        for _, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to(self.params["run_device"])
            labels = labels.to(self.params["run_device"])

            for class_ind in range(class_num):

                indices = labels == class_ind

                samples_per_class[class_ind] = torch.cat(
                    (
                        samples_per_class[class_ind],
                        inputs[indices]
                    ),
                    dim=0
                )

        target_class = self.params[
            "poison_label_swap"
        ][client_id]

        for i in range(class_num):

            sample = samples_per_class[i]

            if len(sample) == 0:
                continue

            outputs = model(sample)

            tmp_label = torch.ones(
                len(outputs),
                dtype=torch.long,
                device=self.params["run_device"]
            ) * i

            loss_sort = criterion(outputs, tmp_label)

            select_len = min(
                len(outputs),
                self.params[
                    "discriminator_train_samples_pre_class"
                ]
            )

            if i == target_class:
                select_len = len(outputs)

            _, indices = torch.topk(
                loss_sort,
                select_len,
                largest=False
            )

            samples_per_class[i] = sample[indices]

        samples_disc = torch.tensor(
            [],
            device=self.params["run_device"]
        )

        labels_disc = torch.tensor(
            [],
            dtype=torch.long,
            device=self.params["run_device"]
        )

        for i in range(class_num):

            if i == target_class:
                continue

            samples = samples_per_class[i]

            labels = torch.ones(
                len(samples),
                dtype=torch.long,
                device=self.params["run_device"]
            )

            poisoned_sample, _ = poisoned_batch_injection(
                (samples, labels),
                trigger=trigger_,
                mask=mask_,
                is_eval=True,
                label_swap=target_class
            )

            samples_disc = torch.cat(
                (samples_disc, poisoned_sample),
                dim=0
            )

            labels_disc = torch.cat(
                (labels_disc, labels),
                dim=0
            )

        samples_disc = torch.cat(
            (
                samples_disc,
                samples_per_class[target_class]
            ),
            dim=0
        )

        labels_disc = torch.cat(
            (
                labels_disc,
                torch.zeros(
                    len(samples_per_class[target_class]),
                    dtype=torch.long,
                    device=self.params["run_device"]
                )
            ),
            dim=0
        )

        discriminator_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                samples_disc,
                labels_disc
            ),
            batch_size=self.params[
                "discriminator_batch_size"
            ],
            shuffle=True
        )

        return discriminator_loader

    # ==========================================================
    # DISCRIMINATOR
    # ==========================================================

    def get_discriminator(
            self,
            model,
            discriminator_dataloader):

        discriminator_ = copy.deepcopy(model)

        if "resnet" in self.params["model_type"].lower():

            discriminator_.linear = torch.nn.Sequential(
                torch.nn.Linear(
                    discriminator_.linear.in_features,
                    10
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 2)
            )

        discriminator_optimizer = torch.optim.SGD(
            discriminator_.parameters(),
            lr=self.params["discriminator_lr"],
            momentum=self.params[
                'discriminator_momentum'
            ],
            weight_decay=self.params[
                'discriminator_weight_decay'
            ]
        )

        discriminator_criterion = nn.CrossEntropyLoss().to(
            self.params["run_device"]
        )

        discriminator_ = discriminator_.to(
            self.params["run_device"]
        )

        discriminator_.train()

        for _ in range(
                self.params[
                    "discriminator_train_no_times"
                ]):

            for batch in discriminator_dataloader:

                inputs, labels = batch

                inputs = inputs.to(self.params["run_device"])
                labels = labels.to(self.params["run_device"])

                outputs = discriminator_(inputs)

                loss = discriminator_criterion(
                    outputs,
                    labels
                )

                discriminator_optimizer.zero_grad()

                loss.backward()

                discriminator_optimizer.step()

        discriminator_.eval()

        return discriminator_

    # ==========================================================
    # AFDM TRIGGER SEARCH
    # ==========================================================

    def search_trigger(
            self,
            model,
            train_loader,
            client_id,
            test_loader=None):

        model.eval()

        mask_ = copy.deepcopy(
            self.mask_set[client_id]
        )

        ce_loss = nn.functional.cross_entropy

        cos_loss = nn.CosineSimilarity(
            dim=1,
            eps=1e-8
        )

        feature_model = self.get_feature_extractor(model)

        freq_t = copy.deepcopy(
            self.freq_trigger_set[client_id]
        )

        freq_t.requires_grad_()

        # ======================================================
        # LOW FREQUENCY MASK
        # ======================================================

        freq_mask = torch.zeros_like(freq_t)

        band_size = self.params[
            "frequency_band_size"
        ]

        freq_mask[:, :band_size, :band_size] = 1

        optimizer = torch.optim.Adam(
            [freq_t],
            lr=self.params["trigger_lr"]
        )

        print(
            "AFDM Trigger Search:",
            self.params["trigger_search_no_times"]
        )

        for _ in tqdm(
                range(
                    self.params[
                        "trigger_search_no_times"
                    ]
                )):

            masked_freq_t = freq_t * freq_mask

            spatial_trigger = self.idct2d(
                masked_freq_t
            )

            discriminator_loader = \
                self.generate_discriminator_dataloader(
                    model,
                    train_loader,
                    spatial_trigger,
                    mask_,
                    client_id
                )

            model_discriminator = self.get_discriminator(
                model,
                discriminator_loader
            )

            for inputs, targets in train_loader:

                inputs = inputs.to(
                    self.params["run_device"]
                )

                targets = targets.to(
                    self.params["run_device"]
                )

                clean_indices = (
                        targets ==
                        self.params[
                            "poison_label_swap"
                        ][client_id]
                )

                if clean_indices.sum() == 0:
                    continue

                backdoor_indices = ~clean_indices

                backdoor_inputs = inputs[
                    backdoor_indices
                ]

                backdoor_targets = targets[
                    backdoor_indices
                ]

                # ==============================================
                # FREQUENCY DOMAIN
                # ==============================================

                freq_inputs = self.dct2d(
                    backdoor_inputs
                )

                feature_statistics = torch.mean(
                    freq_inputs,
                    dim=0
                )

                adaptive_freq = \
                    self.adaptive_frequency_modulation(
                        masked_freq_t,
                        client_id,
                        feature_statistics,
                        freq_mask
                    )

                poisoned_freq = (
                        freq_inputs
                        +
                        adaptive_freq
                )

                backdoor_inputs = self.idct2d(
                    poisoned_freq
                )

                backdoor_inputs = torch.clamp(
                    backdoor_inputs,
                    min=0,
                    max=1
                )

                # ==============================================
                # DISCRIMINATOR LOSS
                # ==============================================

                pred_disc = model_discriminator(
                    backdoor_inputs
                )

                loss_discriminator = ce_loss(
                    pred_disc,
                    torch.zeros(
                        len(pred_disc),
                        device=self.params[
                            "run_device"
                        ]
                    ).long()
                )

                # ==============================================
                # ASR LOSS
                # ==============================================

                backdoor_pred = model(
                    backdoor_inputs
                )

                target_labels = torch.ones_like(
                    backdoor_targets
                ) * self.params[
                    "poison_label_swap"
                ][client_id]

                loss_asr = ce_loss(
                    backdoor_pred,
                    target_labels
                )

                # ==============================================
                # FEATURE SIMILARITY LOSS
                # ==============================================

                clean_features = feature_model(
                    inputs[backdoor_indices]
                )

                poison_features = feature_model(
                    backdoor_inputs
                )

                loss_sim = 1 - cos_loss(
                    poison_features,
                    clean_features
                ).mean()

                # ==============================================
                # SPECTRAL LOSSES
                # ==============================================

                spectral_loss = \
                    self.frequency_smoothness_loss(
                        masked_freq_t
                    )

                clean_freq = self.dct2d(
                    inputs[backdoor_indices]
                )

                poison_freq = self.dct2d(
                    backdoor_inputs
                )

                alignment_loss = \
                    self.spectral_alignment_loss(
                        clean_freq,
                        poison_freq
                    )

                # ==============================================
                # TOTAL LOSS
                # ==============================================

                loss = (
                        loss_discriminator
                        +
                        loss_asr
                        +
                        loss_sim
                        +
                        self.params[
                            "spectral_loss_weight"
                        ] * spectral_loss
                        +
                        self.params[
                            "alignment_loss_weight"
                        ] * alignment_loss
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                freq_t.data = torch.clamp(
                    freq_t.data,
                    min=-1,
                    max=1
                )

        self.freq_trigger_set[client_id] = \
            freq_t.detach()

        final_trigger = self.idct2d(
            freq_t.detach() * freq_mask
        )

        return final_trigger.detach()

    # ==========================================================
    # LOCAL TRAIN
    # ==========================================================

    def local_train(
            self,
            iteration,
            model,
            train_loader,
            client_id,
            test_loader=None):

        cache_model = copy.deepcopy(model)

        optimizer = torch.optim.SGD(
            cache_model.parameters(),
            lr=self.params['poisoned_lr'],
            momentum=self.params[
                'poisoned_momentum'
            ],
            weight_decay=self.params[
                'poisoned_weight_decay'
            ]
        )

        # ======================================================
        # AFDM TRIGGER SEARCH
        # ======================================================

        trigger_ = self.search_trigger(
            cache_model,
            train_loader,
            client_id
        )

        self.trigger_set[client_id] = trigger_

        cache_model.train()

        for epoch in range(
                self.params[
                    "poisoned_retrain_no_times"
                ]):

            total_loss = 0.

            for _, batch in enumerate(train_loader):

                inputs, labels = poisoned_batch_injection(
                    batch,
                    trigger=self.trigger_set[client_id],
                    mask=self.mask_set[client_id],
                    is_eval=False,
                    label_swap=self.params[
                        "poison_label_swap"
                    ][client_id]
                )

                inputs = inputs.to(
                    self.params["run_device"]
                )

                labels = labels.to(
                    self.params["run_device"]
                )

                outputs = cache_model(inputs)

                loss = self.criterion(
                    outputs,
                    labels
                )

                total_loss += loss.item()

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            logger.info(
                f"[AFDM] Epoch {epoch} Loss: {total_loss:.4f}"
            )

        return cache_model