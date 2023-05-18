import torch
from torch.utils.data import Dataset, DataLoader
import cv2

from PIL import Image
import numpy as np
from tqdm import tqdm

from models.mlp import FCNet
from models.pe import PE
from models.grid import DenseNet, InstantNGP

def get_coords(res, normalize = False):
    x = y = torch.arange(res)
    xx, yy = torch.meshgrid(x, y)
    coords = torch.stack([xx, yy], dim=-1)
    if normalize:
        coords = coords / (res - 1)
    return coords

def get_psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr

class ImageDataset(Dataset):
    def __init__(self, image_path, image_size, device = 'cuda'):
        self.image = Image.open(image_path).resize(image_size)
        self.rgb_vals = torch.from_numpy(np.array(self.image)).reshape(-1, 3).to(device)
        self.rgb_vals = self.rgb_vals.float() / 255
        self.coords = get_coords(image_size[0], normalize=True).reshape(-1, 2).to(device)

    def __len__(self):
        return len(self.rgb_vals)
    def __getitem__(self, idx):
        return self.coords[idx], self.rgb_vals[idx]

class Trainer:
    def __init__(self, image_path, image_size, model_type = 'mlp', use_pe = True, device = 'cuda'):
        self.dataset = ImageDataset(image_path, image_size, device)
        self.dataloader = DataLoader(self.dataset, batch_size=4096, shuffle=True)

        if model_type == 'mlp':
           self.model = FCNet().to(device)
        elif model_type == 'dense':
            self.model = DenseNet().to(device)
        elif model_type == 'instant_ngp':
            self.model = InstantNGP().to(device)
        else:
            pass

        lr = 1e-3
        if model_type == 'dense' or model_type == 'instant_ngp':
            lr = 5e-2
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.nepochs = 200

    def run(self):
        pbar = tqdm(range(self.nepochs))
        for epoch in pbar:
            self.model.train()
            for coords, rgb_vals in self.dataloader:
                self.optimizer.zero_grad()
                pred = self.model(coords)
                loss = self.criterion(pred, rgb_vals)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                coords = self.dataset.coords
                pred = self.model(coords)
                gt = self.dataset.rgb_vals
                psnr = get_psnr(pred, gt)
            pbar.set_description(f'Epoch: {epoch}, PSNR: {psnr:.2f}')
            pred = pred.cpu().numpy().reshape(*self.dataset.image.size[::-1], 3)
            pred = (pred * 255).astype(np.uint8)
            gt = self.dataset.rgb_vals.cpu().numpy().reshape(*self.dataset.image.size[::-1], 3)
            gt = (gt * 255).astype(np.uint8)
            save_image = np.hstack([gt, pred])
            save_image = Image.fromarray(save_image)
            #save_image.save(f'output_{epoch}.png')
            self.visualize(np.array(save_image), text = '# params: {}, PSNR: {:.2f}'.format(self.get_num_params(), psnr))

    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def visualize(self, image, text):
        save_image = np.ones((300, 512, 3), dtype=np.uint8) * 255
        img_start = (300 - 256)
        save_image[img_start:img_start + 256, :, :] = image
        save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
        position = (100, 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(save_image, text, position, font, scale, color, thickness)
        cv2.imshow('image', save_image)
        cv2.waitKey(1)



if __name__ == '__main__':
    image_path = 'image.jpg'
    image_size = (256, 256)
    model_type = 'dense'
    device = 'cuda'

    trainer = Trainer(image_path, image_size, model_type, device)
    print('# params: {}'.format(trainer.get_num_params()))
    trainer.run()

