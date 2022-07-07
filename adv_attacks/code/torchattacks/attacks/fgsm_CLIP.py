import torch
import torch.nn as nn

from ..attack import Attack
import clip


class FGSM_CLIP(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.007)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=0.007)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.007):
#         print("Inside init")
#         print("Inside Saqib's init!!!")
        super().__init__("FGSM_CLIP", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels, text_features=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        images.requires_grad = True
#         loss = nn.CrossEntropyLoss()
        
        img_features = self.model.encode_image(images).float()
        img_features = img_features / torch.linalg.norm(img_features, dim=1, keepdim=True)
        
        similarity_map = torch.matmul(img_features, text_features.T)
        print(similarity_map.shape, labels.shape)
        
#         outputs = self.model(images)

        # Calculate loss
#         if self._targeted:
#             cost = -loss(outputs, target_labels)
#         else:
#             cost = loss(outputs, labels)

        # Update adversarial images
#         grad = torch.autograd.grad(cost, images,
#                                    retain_graph=False, create_graph=False)[0]

#         adv_images = images + self.eps*grad.sign()
#         adv_images = torch.clamp(adv_images, min=0, max=1).detach()

#         return adv_images
        return 0
