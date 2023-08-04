import torch
import torchvision
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics
import wandb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, classification_report, auc, ConfusionMatrixDisplay


class VideoClassificationLightningModule(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()

        self.args = args
        self.model = model
        self.dataloader_length = 0
        self.classes = ['Feeding', 'Grooming', 'Pumping']

        self.save_hyperparameters("args")
        
        # For logging outputs
        self.epoch_logits = []
        self.epoch_incorrect_samples = set()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, stage='val')
    
    def _common_step(self, batch, batch_idx, stage='train'):
        X, y = batch['video'], batch['label']

        output = self(X.permute(0, 2, 1, 3, 4)) # (8, 3, 16, 224, 224) -> (8, 16, 3, 224, 224)

        loss = F.cross_entropy(output.logits, y)
        acc = torchmetrics.functional.accuracy(output.logits, y, task="multiclass", num_classes=3)

        self.log(
            f"{stage}_loss", loss.item(), batch_size=self.args.batch_size, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            f"{stage}_acc", acc, batch_size=self.args.batch_size, on_step=False, on_epoch=True, prog_bar=True
        )
        if stage == 'train':
            return loss
        elif stage == 'val':
            predictions = torch.argmax(output.logits, dim=1)
            incorrect_indices = torch.nonzero(predictions != y).squeeze()  # Get indices of incorrect samples
            incorrect_indices = incorrect_indices.tolist()

            if isinstance(incorrect_indices, int):
                incorrect_indices = [incorrect_indices]
            incorrect_samples = [batch['video_name'][idx] for idx in incorrect_indices]
            self.epoch_logits.extend(output.logits)
            for i in list(set(incorrect_samples)):
                self.epoch_incorrect_samples.add(i)
            

        
    def on_validation_epoch_end(self):
        #dummy_input = torch.zeros((1, 8, 3, 224, 224), device=self.device)
        #model_filename = "model_ckpt.onnx"
        #torch.onnx.export(self, dummy_input, model_filename, opset_version=11)
        #artifact = wandb.Artifact(name="model.ckpt", type="model")
        #artifact.add_file(model_filename)
        #self.logger.experiment.log_artifact(artifact)

        flattened_logits = torch.flatten(torch.cat(self.epoch_logits))
        
        incorrect_sample_df = pd.DataFrame({'false_predictions':list(self.epoch_incorrect_samples)})
        self.logger.log_text(key="incorrect_preds", dataframe=incorrect_sample_df)
        self.logger.experiment.log(
            {"valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
             "global_step": self.global_step})

        self.epoch_logits.clear()
        self.epoch_incorrect_samples = set()
        


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, y = batch['video'], batch['label']
        output = self(X.permute(0, 2, 1, 3, 4))

        # Convert the predicted probabilities to class predictions
        y_pred = torch.argmax(output.logits, dim=1)
        y_prob = torch.softmax(output.logits, dim=1)
        y_probs = y_prob.cpu().numpy()

        if batch_idx == 0:
            self.all_y = y.cpu().numpy()
            self.all_y_probs = y_probs
        else:
            self.all_y = np.concatenate([self.all_y, y.cpu().numpy()])
            self.all_y_probs = np.concatenate([self.all_y_probs, y_probs])


        # Perform the computations for the last batch
        if batch_idx == self.dataloader_length - 1:
            # Compute the ROC curve
            fprs = []
            tprs = []
            for i in range(len(self.classes)):
                fpr, tpr, threshold = roc_curve((self.all_y == i), self.all_y_probs[:, i])
                fprs.append(fpr)
                tprs.append(tpr)

            self._plot_roc_curve(fprs, tprs)

            # Generate the confusion matrix
            cm = confusion_matrix(self.all_y, np.argmax(self.all_y_probs, axis=1))

            # Plot the confusion matrix
            self._plot_confusion_matrix(cm)

            # Compute the classification report
            report = classification_report(self.all_y, np.argmax(self.all_y_probs, axis=1), target_names=self.classes)
            print(report)

        return output.logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.lr,
        )
        return [optimizer]

    def _plot_roc_curve(self, fprs, tprs):
        # Plot the ROC curve for each class
        plt.figure()
        for i in range(len(self.classes)):
            plt.plot(fprs[i], tprs[i], label=f'Class {self.classes[i]}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig('fly_roc.png')

    def _plot_confusion_matrix(self, cm):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=self.classes)
        disp.plot()
        plt.savefig('fly_cm.png')