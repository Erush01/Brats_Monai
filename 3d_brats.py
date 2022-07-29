import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism
import torch


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC(Tumor core), WT(Whole tumor)
    and ET(Enhancing tumor)
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))

            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()

        return d

if __name__ == "__main__":
    os.environ['MONAI_DATA_DIRECTORY'] = ""
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = directory

    set_determinism(0)



    train_transform = Compose(
        [
            #load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode = ("bilinear", "nearest")
            ),
            RandSpatialCropd(keys=["image", "label"],roi_size=[224,224,144],random_size=False),
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image",nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image",factors=0.1,prob=0.1),
            RandShiftIntensityd(keys="image",offsets=0.1,prob=1.0)

        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image","label"]),
            EnsureChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image","label"],axcodes="RAS"),
            Spacingd(
                keys=["image","label"],
                pixdim = (1.0, 1.0, 1.0),
                mode = ("bilinear","nearest")
            ),
            NormalizeIntensityd(keys="image",nonzero=True, channel_wise=True)
        ]
    )
    train_ds = DecathlonDataset(
        root_dir= root_dir,
        task = "Task01_BrainTumour",
        transform=train_transform,
        section="training",
        download=True,
        cache_rate=0.0,
        num_workers=2
    )
    train_loader = DataLoader(train_ds, batch_size=1,shuffle=True, num_workers=2)
    val_ds = DecathlonDataset(
        root_dir = root_dir,
        task="Task01_BrainTumour",
        transform=val_transform,
        section="validation",
        download=False,
        cache_rate=0.0,
        num_workers=2
    )
    val_loader = DataLoader(val_ds,batch_size=1, shuffle=False, num_workers=2)

    val_data_example = val_ds[2]
    print(f"image shape: {val_data_example['image'].shape}")
    plt.figure("image",(24,6))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.title(f"image channel {i}")
        plt.imshow(val_data_example["image"][i, :, :,60].detach().cpu(), cmap = "gray")
    plt.show()
    print(f"label shape: {val_data_example['label'].shape}")
    plt.figure("label", (18,6))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(f"label channel {i}")
        plt.imshow(val_data_example["label"][i, :, :,60].detach().cpu())

    plt.show()

    max_epochs = 1
    val_interval = 1
    VAL_AMP = True

    device = torch.device("cuda")
    model = SegResNet(
        blocks_down = [1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2
    ).to(device)
    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    post_trans = Compose(
        [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
    )

    #define inference method
    def inference(input):

        def _compute(input):
            return sliding_window_inference(
                inputs= input,
                roi_size=(240,240,160),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5
            )
        if VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)

    #use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    #enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    #**************TRAINING*******************#
    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    total_start = time.time()
    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-"*10)
        print(f"epoch {epoch+1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device)
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs,labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}"
                f", train_loss: {loss.item():.4f}"
                f", step time: {(time.time()- step_start):.4f}")

        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch+1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device))

                    val_outputs = inference(val_inputs)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = dice_metric_batch.aggregate()
                metric_tc = metric_batch[0].item()
                metric_values_tc.append(metric_tc)
                metric_wt = metric_batch[1].item()
                metric_values_wt.append(metric_wt)
                metric_et = metric_batch[2].item()
                metric_values_et.append(metric_et)
                dice_metric.reset()
                dice_metric_batch.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                    print("saved new best metric model")
                    print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                            f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                            f"\nbest mean dice: {best_metric:.4f}"
                            f" at epoch: {best_metric_epoch}")

        print(f"time consuming of epoch {epoch +1 } is: {(time.time() - epoch_start):.4f} ")
    total_time = time.time() - total_start
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="green")
    plt.show()

    plt.figure("train", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Val Mean Dice TC")
    x = [val_interval * (i + 1) for i in range(len(metric_values_tc))]
    y = metric_values_tc
    plt.xlabel("epoch")
    plt.plot(x, y, color="blue")
    plt.subplot(1, 3, 2)
    plt.title("Val Mean Dice WT")
    x = [val_interval * (i + 1) for i in range(len(metric_values_wt))]
    y = metric_values_wt
    plt.xlabel("epoch")
    plt.plot(x, y, color="brown")
    plt.subplot(1, 3, 3)
    plt.title("Val Mean Dice ET")
    x = [val_interval * (i + 1) for i in range(len(metric_values_et))]
    y = metric_values_et
    plt.xlabel("epoch")
    plt.plot(x, y, color="purple")
    plt.show()
    model.load_state_dict(
        torch.load(os.path.join(root_dir,"best_metric_model.pth")))
    model.eval()
    with torch.no_grad():
        val_input = val_ds[6]["image"].unsqueeze(0).to(device)
        roi_size = (128, 128, 64)
        sw_batch_size = 4
        val_output = inference(val_input)
        val_output = post_trans(val_output[0])
        plt.figure("image", (24,6))
        for i in range(4):
            plt.subplot(1, 4, i+1)
            plt.title(f"image channel {i}")
            plt.imshow(val_ds[6]["image"][i, :, :, 70].detach().cpu(), cmap="gray")
        plt.imshow()
        plt.figure("label", (18,6))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(f"label channel {i}")
            plt.imshow(val_ds[6]["label"][i, :, :, 70].detach().cpu())
        plt.show()
        plt.figure("output", (18,6))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(f"output channel {i}")
            plt.imshow(val_output[i, :, :, 70].detach().cpu())
        plt.show()
    val_org_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    val_org_ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        transform=val_org_transforms,
        section="validation",
        download=False,
        num_workers=4,
        cache_num=0,
    )
    val_org_loader = DataLoader(val_org_ds, batch_size=1, shuffle=False, num_workers=4)

    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=val_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5),
    ])
    model.load_state_dict(torch.load(
        os.path.join(root_dir, "best_metric_model.pth")))
    model.eval()

    with torch.no_grad():
        for val_data in val_org_loader:
            val_inputs = val_data["image"].to(device)
            val_data["pred"] = inference(val_inputs)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            dice_metric(y_pred=val_outputs, y=val_labels)
            dice_metric_batch(y_pred=val_outputs, y=val_labels)

        metric_org = dice_metric.aggregate().item()
        metric_batch_org = dice_metric_batch.aggregate()

        dice_metric.reset()
        dice_metric_batch.reset()

    metric_tc, metric_wt, metric_et = metric_batch[0].item(), metric_batch[1].item(), metric_batch[2].item()

    print("Metric on original image spacing: ", metric)
    print(f"metric_tc: {metric_tc:.4f}")
    print(f"metric_wt: {metric_wt:.4f}")
    print(f"metric_et: {metric_et:.4f}")