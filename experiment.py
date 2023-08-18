import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
from torch import nn

# files = os.listdir(root)
# # for i, file in enumerate(files[0:3]):

# # image_dir = os.path.join(root,file)

# # print(image_dir)
# sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
# sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)

# sobel_x = sobel_x.view(1, 1, 3, 3)
# sobel_y = sobel_y.view(1, 1, 3, 3)

# gradient_x = F.conv2d(image_tensor, sobel_x)
# gradient_y = F.conv2d(image_tensor, sobel_y)

# gradient = gradient_x + gradient_y



class FLT_head(nn.Module):
    def __init__(self):
        super(FLT_head,self).__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x
    
if __name__ == '__main__':
    root = 'datasets/satellite102/trainB/'
    # image = Image.open(root + 'SAR_8bit_ver1_ROI_195.TIF')
    image = Image.open(root + 'k3a_L_1007_8bit_ROI_9.TIF')
    transform = transforms.ToTensor() 
    image_tensor = transform(image) 
    image_tensor = torch.unsqueeze(image_tensor, 0)
    print(image_tensor.shape)
    flt_head = FLT_head()
    output = flt_head(image_tensor)
    output = torch.squeeze(output)
    transform2 = transforms.ToPILImage()
    image_pil = transform2(output)
    print(np.array(image_pil))
    print(np.array(image_pil).shape)
    image_pil.save('experiment_y','png')