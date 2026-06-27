import pandas as pd
import torch


class Trainer:
    def __init__(
        self,
        data_loaders,
        criterion,
        device,
        scheduler=None,
        on_after_epoch=None,
        on_before_epoch=None,
    ):
        self.data_loaders = data_loaders
        self.criterion = criterion
        self.device = device
        self.history = []
        self.on_after_epoch = on_after_epoch
        self.on_before_epoch = on_before_epoch
        self.scheduler = scheduler

    def train(self, model, optimizer, num_epochs):
        for epoch in range(num_epochs):

            if self.on_before_epoch is not None:
                self.on_before_epoch(epoch)

            train_stats = self._train_on_epoch(model, optimizer)
            val_stats = self._val_on_epoch(model)

            if self.scheduler is not None:

                self.scheduler.step()

            hist = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_bce": train_stats["loss_bce"],
                "train_dice": train_stats["loss_dice"],
                "train_centroid": train_stats["loss_centroid"],
                "val_loss": val_stats["loss"],
                "val_bce": val_stats["loss_bce"],
                "val_dice": val_stats["loss_dice"],
                "val_centroid": val_stats["loss_centroid"],
                "current_lr": round(optimizer.param_groups[0]["lr"], 8),
            }
            self.history.append(hist)

            if self.on_after_epoch is not None:
                self.on_after_epoch(model, pd.DataFrame(self.history))

        return pd.DataFrame(self.history)

    def _train_on_epoch(self, model, optimizer):
        model.train()
        data_loader = self.data_loaders[0]

        running = {"loss": 0.0, "loss_bce": 0.0, "loss_dice": 0.0, "loss_centroid": 0.0}
        n_seen = 0

        for batch in data_loader:
            inputs = batch["image"].to(self.device)
            labels = batch["mask"].to(self.device)
            centers_gt = batch["centers_gt"].to(self.device)
            centers_valid = batch["centers_valid"].to(self.device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss_dict = self.criterion(outputs, labels, centers_gt, centers_valid)
                loss = loss_dict["loss"]
                loss.backward()
                optimizer.step()

            bs = inputs.size(0)
            n_seen += bs
            for k in running.keys():
                running[k] += loss_dict[k].item() * bs

        n_seen = max(1, n_seen)
        for k in running.keys():
            running[k] /= n_seen

        return running

    def _val_on_epoch(self, model):
        model.eval()
        data_loader = self.data_loaders[1]

        running = {"loss": 0.0, "loss_bce": 0.0, "loss_dice": 0.0, "loss_centroid": 0.0}
        n_seen = 0

        for batch in data_loader:
            inputs = batch["image"].to(self.device)
            labels = batch["mask"].to(self.device)
            centers_gt = batch["centers_gt"].to(self.device)
            centers_valid = batch["centers_valid"].to(self.device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss_dict = self.criterion(outputs, labels, centers_gt, centers_valid)

            bs = inputs.size(0)
            n_seen += bs
            for k in running.keys():
                running[k] += loss_dict[k].item() * bs

        n_seen = max(1, n_seen)
        for k in running.keys():
            running[k] /= n_seen

        return running
