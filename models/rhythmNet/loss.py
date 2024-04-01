import torch
import torch.nn as nn


class MyLoss(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, hr_t, hr_outs, T):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.hr_outs = hr_outs
        ctx.hr_mean = hr_outs.mean()
        ctx.T = T
        ctx.save_for_backward(hr_t)
        # pdb.set_trace()
        # hr_t, hr_mean, T = input

        if hr_t > ctx.hr_mean:
            loss = hr_t - ctx.hr_mean
        else:
            loss = ctx.hr_mean - hr_t

        return loss
        # return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        output = torch.zeros(1).to(torch.device("cuda"))

        hr_t, = ctx.saved_tensors
        hr_outs = ctx.hr_outs

        # create a list of hr_outs without hr_t

        for hr in hr_outs:
            if hr == hr_t:
                pass
            else:
                output = output + (1/ctx.T)*torch.sign(ctx.hr_mean - hr)

        output = (1/ctx.T - 1)*torch.sign(ctx.hr_mean - hr_t) + output

        return output, None, None



class RhythmNetLoss(nn.Module):
    def __init__(self, weight=100.0):
        super(RhythmNetLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.lambd = weight
        self.gru_outputs_considered = None
        self.custom_loss = MyLoss()
        self.device = torch.device("cuda")

    def forward(self, resnet_outputs, gru_outputs, target):
        frame_rate = 25.0
        # resnet_outputs, gru_outputs, _ = outputs
        # target_array = target.repeat(1, resnet_outputs.shape[1])
        l1_loss = self.l1_loss(resnet_outputs, target)
        smooth_loss_component = self.smooth_loss(gru_outputs)

        loss = l1_loss + self.lambd*smooth_loss_component
        return loss

    # Need to write backward pass for this loss function
    def smooth_loss(self, gru_outputs):
        smooth_loss = torch.zeros(1).to(device=self.device)
        self.gru_outputs_considered = gru_outputs.flatten()
        # hr_mean = self.gru_outputs_considered.mean()
        for hr_t in self.gru_outputs_considered:
            # custom_fn = MyLoss.apply
            smooth_loss = smooth_loss + self.custom_loss.apply(torch.autograd.Variable(hr_t, requires_grad=True),
                                                               self.gru_outputs_considered,
                                                               self.gru_outputs_considered.shape[0])
        return smooth_loss / self.gru_outputs_considered.shape[0]
