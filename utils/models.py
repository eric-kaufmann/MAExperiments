import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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

class ConvVAE(nn.Module):
    def __init__(self, batch_size, grid=64, cond_dim=0):
        super(ConvVAE, self).__init__()
        self.batch_size = batch_size
        self.grid = grid
        
        self.flatten = nn.Flatten()
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.after_cond_encoder = nn.Sequential(
            nn.Linear(131072+cond_dim, 512),
            nn.ReLU()
        )
        self.pre_cond_decoder = nn.Sequential(
            nn.Linear(256+cond_dim, 131072),
            nn.ReLU()
        )
        
    def get_encoder(self):
        if self.grid == 64:
            encoder_header = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm3d(16),
            )
        else:
            encoder_header = nn.Sequential(
                nn.Conv3d(1, 3, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv3d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm3d(16),
            )    
        encoder = nn.Sequential(
            encoder_header,
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(192),
            nn.Conv3d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(256),
        )
        return encoder
    
    def get_decoder(self):
        if self.grid == 64:
            decoder_footer = nn.Sequential(
                nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(3),
            )
        else:
            decoder_footer = nn.Sequential(
                nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(3),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(3, 3, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(3),
            )
                

        decoder = nn.Sequential(
            nn.Conv3d(256, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            decoder_footer
        )
        return decoder

    def forward(self, x, cond_tensor=torch.tensor([])):
       
        # encoder
        x = self.encoder(x.unsqueeze(1))
        
        # add conditioning variables to feature vector
        x = torch.concat([self.flatten(x), cond_tensor], axis=1)
        
        # put new feature vector through the after_cond_encoder
        x = self.after_cond_encoder(x)
        
        # reparameterize
        mu, log_var = torch.chunk(x, 2, dim=1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # add conditioning variables to z
        z = torch.concat([z, cond_tensor], axis=1)
        
        # put z through the pre_cond_decoder
        z = self.pre_cond_decoder(z)
        
        # reshape z to block shape
        z = torch.reshape(z, (self.batch_size, 256, 8, 8, 8))
        
        # put new block shaped z through the decoder
        z = F.interpolate(z, scale_factor=2, mode='trilinear', align_corners=False)
        z = self.decoder[0:3](z)
        z = self.decoder[3:6](z)
        z = F.interpolate(z, scale_factor=2, mode='trilinear', align_corners=False)
        z = self.decoder[6:9](z)
        z = self.decoder[9:12](z)
        z = F.interpolate(z, scale_factor=2, mode='trilinear', align_corners=False)
        z = self.decoder[12:](z)
        return z, mu, log_var