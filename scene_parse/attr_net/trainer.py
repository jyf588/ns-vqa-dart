import os
import json
import torch
import time


class Trainer:
    def __init__(self, opt, model, train_loader, val_loader=None):
        self.num_iters = opt.num_iters
        self.run_dir = opt.run_dir
        self.display_every = opt.display_every
        self.checkpoint_every = opt.checkpoint_every

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model

        self.checkpoint_t = opt.checkpoint_t

        if opt.load_checkpoint_path is not None:
            self.stats = json.load(open(f"{self.run_dir}/stats.json"))
            assert opt.checkpoint_t is not None
        else:
            self.stats = {
                "train_losses": [],
                "train_losses_ts": [],
                "val_losses": [],
                "val_losses_ts": [],
                "best_val_loss": 9999,
                "model_t": 0,
                "old_losses": [],
                "rot_losses": [],
            }

        self.opt = opt

    def train(self):
        print("| start training, running in directory %s" % self.run_dir)
        t = 0 if self.checkpoint_t is None else self.checkpoint_t
        epoch = 0
        start_time = time.time()
        while t < self.num_iters:
            epoch += 1
            for data, label, _, _ in self.train_loader:
                t += 1
                self.model.set_input(data, label)
                self.model.step()
                loss = self.model.get_loss()

                if t % self.display_every == 0:
                    self.stats["train_losses"].append(loss)
                    avg_iter_time = (time.time() - start_time) / t
                    print(
                        "| iteration %d / %d, epoch %d, loss %f, avg_iter_secs: %.2f\n"
                        % (t, self.num_iters, epoch, loss, avg_iter_time),
                        end="",
                    )
                    self.stats["train_losses_ts"].append(t)

                if t % self.checkpoint_every == 0 or t >= self.num_iters:
                    if self.val_loader is not None:
                        print("| checking validation loss")
                        val_loss = self.check_val_loss()
                        print("| validation loss %f" % val_loss)
                        if val_loss <= self.stats["best_val_loss"]:
                            print("| best model")
                            self.stats["best_val_loss"] = val_loss
                            self.stats["model_t"] = t
                            self.model.save_checkpoint(
                                "%s/checkpoint_best.pt" % self.run_dir
                            )
                        self.stats["val_losses"].append(val_loss)
                        self.stats["val_losses_ts"].append(t)
                    print("| saving checkpoint")
                    self.model.save_checkpoint(
                        "%s/checkpoint_iter%08d.pt" % (self.run_dir, t)
                    )
                    self.model.save_checkpoint(
                        os.path.join(self.run_dir, "checkpoint.pt")
                    )
                    with open("%s/stats.json" % self.run_dir, "w") as fout:
                        json.dump(self.stats, fout)

                if t >= self.num_iters:
                    break

    def check_val_loss(self):
        self.model.eval_mode()
        loss = 0
        t = 0
        for x, y, _, _ in self.val_loader:
            self.model.set_input(x, y)
            self.model.forward()
            loss += self.model.get_loss()
            t += 1
        self.model.train_mode()
        return loss / t if t is not 0 else 0


def get_trainer(opt, model, train_loader, val_loader=None):
    return Trainer(opt, model, train_loader, val_loader)
