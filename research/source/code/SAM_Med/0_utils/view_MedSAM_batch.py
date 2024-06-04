from torchvision.utils import make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def display_batch(batch, show_points = True, show_boxes = True, size=20):
    # get the shape of the batch
    n = batch['image'].shape[0]
    # we plot the grid as a square; therefore, construct a square grid
    nrows = int(n ** 0.5)

    images = F.interpolate(batch['image'], size=(256, 256), mode='bilinear', align_corners=False)

    grid_imgs = make_grid(images, nrow=nrows, padding=0)
    grid_gts = make_grid(batch['gt2D'].float(), nrow=nrows, padding=0)
    gts_mask = (grid_gts.sum(dim=0) > 0).float()

    plt.figure(figsize=(size, size))
    plt.imshow(grid_imgs.permute(1, 2, 0))
    plt.imshow(gts_mask, alpha=gts_mask, cmap='viridis')

    shift_x = 0
    shift_y = -256
    for i in range(n):

        shift_y = shift_y + 256 if i % nrows == 0 else shift_y
        shift_x = shift_x + 256 if i % nrows != 0 else 0

        if show_boxes:
            coord = batch['boxes'][i, 0].numpy().astype(np.uint16)
            for c in coord:
                x_min, y_min, x_max, y_max = c[0], c[1], c[2], c[3]
                
                # plot the box
                x, y = x_min, y_min
                x, y = x * 256 / 1024 + shift_x, y * 256 / 1024 + shift_y

                h, w = y_max - y_min, x_max - x_min
                h, w = h * 256 / 1024, w * 256 / 1024

                rectangle = plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rectangle)

        if show_points:
            coord = batch['coords'][i].squeeze().numpy()
            for c in coord:
                x, y = c[0], c[1]
                x, y = x * 256 / 1024 + shift_x, y * 256 / 1024 + shift_y
                plt.scatter(x, y, c='r', s=60)

    plt.axis('off')
    plt.show()