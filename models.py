import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaHAN(nn.Module):
    """
    A variant of the Adaptive Hard Attention Network presented in the 
    "Learning Visual Question Answering by Bootstrapping Hard Attention" 
    research paper, by Mateusz Malinowski, Carl Doersch, Adam Santoro, 
    and Peter Battaglia of DeepMind, London.

    Here we omit the LSTM embedding as I am only trying to use this for image classification
    """

    def __init__(self, hidden_size, n_classes, k=2, adaptive=False):
        super(AdaHAN, self).__init__()
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.k = k
        self.adaptive = adaptive

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, padding=2, stride=1),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=5, padding=2, stride=1),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=5, padding=2, stride=1),
            # nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )
        self.conv1by1 = nn.Conv2d(hidden_size, 2, kernel_size=1, padding=0, stride=1)  
        self.fc1 = nn.Linear(24**2, n_classes)
        # there is a way to get the size of the output instead of doing 24**2

    def forward(self, image_tensor):
        x = self.encoder_cnn(image_tensor)
        m = self.conv1by1(x).squeeze(dim=0)
        if len(m) < 4:
            m = m.unsqueeze(0)
        p = torch.sum(m ** 2, dim=1).view(-1, 24**2)
        topk = torch.topk(p, k=self.k)[1]
        m = torch.sum(m, dim=1).view(-1, 24**2)
        
        if self.adaptive:
            latent_mask = torch.where(F.softmax(p, dim=0) >= (1/len(m)), 
                                      torch.ones_like(p), torch.zeros_like(p))  
        else:
            latent_mask = torch.zeros_like(p)
            for i in range(latent_mask.shape[0]):
                latent_mask[i][topk[i]] = 1
        
        # zeros out the activations at masked spatial locations
        attended_vector = m * latent_mask  
        pred = F.log_softmax(self.fc1(attended_vector), dim=1)
        
        return pred, latent_mask


class AttentionVisualizer(nn.Module):
    ''' A class that accepts a latent mask, and uses a dummy input variable
        passed through convolutional layers of identical kernel size,
        padding, and stride as those of the EncoderCNN in HAN and AdaHAN.
        By backpropagating the latent output to the dummy input, we can pinpoint
        the locations in the original input image that correspond to spatial locations in the latent,
        which allows us to visualize binary attention masks.
    '''
    def __init__(self):
        super(AttentionVisualizer, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        self.conv1 = nn.Conv2d(3, 1, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=5, padding=2, stride=1)
        # self.conv3 = nn.Conv2d(1, 1, kernel_size=5, padding=2, stride=1)
        
        self.dummy_input = torch.ones(1, 3, 96, 96, requires_grad=True)


    def forward(self, input_image, latent_mask):
        self.zero_grad()  # reset the gradients
        x = self.pool(self.conv1(self.dummy_input))
        x = self.pool(self.conv2(x))
        # x = self.pool(self.conv3(x))
        x = torch.sum(x.view(-1) * latent_mask)
        x.backward()
        
        n_selected_pixels = torch.sum(torch.where(self.dummy_input.grad != 0, torch.ones_like(input_image), torch.zeros_like(input_image))).item()
        
        print("Number of latent spatial locations selected for attention: {}.".format(torch.sum(latent_mask).item()))
        print("Percentage of input image pixels attended to: {:.1f}%.\n".format(n_selected_pixels/(3 * 96 * 96) * 100))

        attended_image = torch.where(self.dummy_input.grad != 0, input_image, torch.ones_like(input_image))  # this whites out the unattended spatial locations
        return attended_image