import torch
import numpy as np
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No ReLU after the last layer
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

        
class Encoder(nn.Module):
    def __init__(self, layer_sizes):
        super(Encoder, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No ReLU after the last layer
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class EncoderMLP(nn.Module):
    def __init__(self, encoder, mlp):
        super(EncoderMLP, self).__init__()
        self.encoder = encoder
        self.mlp = mlp

    def forward(self, point, geometry):
        return self.mlp(torch.cat([point, self.encoder(geometry.flatten(start_dim=1))], dim=1))

        
class PointNetEncoder(nn.Module):
    def __init__(self, in_channels, z_size, use_bias=True):
        super().__init__()

        self.z_size = z_size
        self.use_bias = use_bias
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.num_classes = 2
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=self.num_classes, bias=self.use_bias),
        ) 
        self.embedding_layer = nn.Embedding(self.num_classes, self.z_size)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, bias=self.use_bias),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True)
        )

        self.mu_layer = nn.Linear(256, self.z_size, bias=True)
        self.std_layer = nn.Linear(256, self.z_size, bias=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        logit = self.fc(output2)
        mu = self.mu_layer(logit)
        logvar = self.std_layer(logit)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class PointNetDecoder(nn.Module):
    def __init__(self, z_size, use_bias=True, out_dim=128**3):
        super().__init__()

        self.z_size = z_size
        self.use_bias = use_bias 
        self.out_dim = out_dim

        self.model = nn.Sequential(            
            nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=256, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=256, out_features=self.out_dim * 3, bias=self.use_bias),
        )

    def forward(self, input):
        z, mu, logvar = input
        output = self.model(z.squeeze())
        output = output.view(-1, self.out_dim, 3)
        return output
    
class PointNet(nn.Module):
    def __init__(self, encoder, decoder, sigma_scaling=0.0, device='cpu'):
        super(PointNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sigma_scaling = sigma_scaling
        self.device = device

    def forward(self, input_tensor):
        z, mu, log_var = self.encoder(input_tensor.permute(0, 2, 1))
        
        sigmas = torch.exp(log_var * 0.5).to(self.device)
        if self.sigma_scaling > 0.0:
            z_gold = torch.randn(self.z_size).to(self.device) * sigmas * self.sigma_scaling + mu
        else:
            z_gold =  mu   
        
        x_hat = self.decoder(
            (z_gold, mu, log_var)
        )
        
        return x_hat
