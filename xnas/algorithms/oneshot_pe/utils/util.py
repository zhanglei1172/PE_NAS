import functools
import torch
import torch.nn as nn

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        """
        Initialization of AverageMeter
        Parameters
        ----------
        name : str
            Name to display.
        fmt : str
            Format string to print the values.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)



def set_running_statistics(configs, model, data_loader):
    bn_mean = {}
    bn_var = {}

    def bn_forward_hook(bn, inputs, outputs, mean_est, var_est):
        aggregate_dimensions = (0, 2, 3)
        inputs = inputs[0]  # input is a tuple of arguments
        batch_mean = inputs.mean(aggregate_dimensions, keepdim=True)  # 1, C, 1, 1
        batch_var = (inputs - batch_mean) ** 2
        batch_var = batch_var.mean(aggregate_dimensions, keepdim=True)

        batch_mean = torch.squeeze(batch_mean)
        batch_var = torch.squeeze(batch_var)

        mean_est.update(batch_mean.data, inputs.size(0))
        var_est.update(batch_var.data, inputs.size(0))

    handles = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_mean[name] = AverageMeter("mean")
            bn_var[name] = AverageMeter("var")
            handle = m.register_forward_hook(functools.partial(bn_forward_hook,
                                                               mean_est=bn_mean[name],
                                                               var_est=bn_var[name]))
            handles.append(handle)

    model.train()
    with torch.no_grad():
        for i in range(configs.bn_sanitize_steps):
            images, _ = next(data_loader)
            images = images.cuda()
            model(images)
        # set_dynamic_bn_running_stats(False)

        for name, m in model.named_modules():
            if name in bn_mean and bn_mean[name].count > 0:
                feature_dim = bn_mean[name].avg.size(0)
                assert isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
                m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
                m.running_var.data[:feature_dim].copy_(bn_var[name].avg)

    for handle in handles:
        handle.remove()
