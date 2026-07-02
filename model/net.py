"""The network that tells apart the two winding directions of a spiral galaxy.

The big idea, in plain words
----------------------------
A spiral galaxy's arms wind either like the letter S or like the letter Z.
If you look at a photo of an S galaxy in a mirror, you see a Z galaxy.
Everything else about the photo stays the same kind of thing in the mirror.
The blob in the middle still looks like a blob. The noise still looks like
noise. A tilted disk still looks like a tilted disk.

We use that fact directly. The network is an ordinary small CNN that gives
each image one score. To classify a galaxy, we run the SAME network twice,
once on the image and once on its mirror image, and we subtract:

    answer = score(image) - score(mirror image)

If the answer is positive we say Z (clockwise). If it is negative we say
S (counterclockwise). This subtraction has a useful property. Any feature
that looks the same in the mirror contributes the same amount to both
scores, so it cancels out of the answer. The winding direction is the one
thing that does NOT cancel, because the mirror reverses it. So the network
can only make its decision from the winding direction. It cannot cheat by
looking at brightness, size, tilt, or noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# A fixed 3x3 blur. The numbers come from the pattern [1,2,1] in both
# directions, scaled so they sum to 1. We blur before shrinking an image so
# that fine details do not turn into sharp artifacts.
_BLUR = torch.tensor([[1.0, 2.0, 1.0],
                      [2.0, 4.0, 2.0],
                      [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 16.0


class BlurAndDownsample(nn.Module):
    """Blur the image slightly, then keep every second pixel.

    This halves the width and height. The blur step means we average
    neighboring pixels instead of just throwing three out of four away,
    which keeps the result smooth. There are no trainable parameters.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("kernel", _BLUR.clone())

    def forward(self, x):
        channels = x.shape[1]
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(x, self.kernel.expand(channels, 1, 3, 3),
                        stride=2, groups=channels)


class SpiralEncoder(nn.Module):
    """A small CNN that turns one grayscale image into one number.

    The structure is three rounds of the same recipe. Each round slides a
    set of small learned filters over the image (a convolution), rescales
    the results so the numbers stay in a healthy range for training (a
    normalization layer), and keeps only the positive parts (the ReLU
    function, which lets the network build up nonlinear decisions).
    Between rounds we shrink the image so later filters see bigger
    structures. At the end we average each filter's map down to a single
    number and combine those numbers into one score with a small linear
    layer. With width 16 the whole network has 23,073 learned numbers.
    """

    def __init__(self, width=16):
        super().__init__()
        w = width
        self.conv1 = nn.Conv2d(1, w, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.GroupNorm(min(4, w), w)
        self.conv2 = nn.Conv2d(w, 2 * w, kernel_size=5, padding=2)
        self.norm2 = nn.GroupNorm(min(4, 2 * w), 2 * w)
        self.conv3 = nn.Conv2d(2 * w, 2 * w, kernel_size=3, padding=1)
        self.norm3 = nn.GroupNorm(min(4, 2 * w), 2 * w)
        self.shrink1 = BlurAndDownsample()
        self.shrink2 = BlurAndDownsample()
        self.head = nn.Linear(2 * w, 1)

    def forward(self, x):
        x = self.shrink1(F.relu(self.norm1(self.conv1(x))))
        x = self.shrink2(F.relu(self.norm2(self.conv2(x))))
        x = F.relu(self.norm3(self.conv3(x)))
        x = x.mean(dim=(2, 3))          # average each filter map to one number
        return self.head(x).squeeze(-1) # combine into one score per image


class MirrorDifferenceNet(nn.Module):
    """Computes the difference of the image score and the mirror image score.

    Both scores come from the same SpiralEncoder with the same weights. The
    output is positive for clockwise and negative for counterclockwise 
    handedness. Mirroring the input flips the sign of the outputexactly, 
    by construction. Features that have no chirality, like a spheroidal 
    bulge component, or random noise, should approximately zero out.
    """

    def __init__(self, width=16):
        super().__init__()
        self.score = SpiralEncoder(width)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        mirrored = torch.flip(x, dims=(-1,))
        return self.score(x) - self.score(mirrored)


if __name__ == "__main__":
    net = MirrorDifferenceNet()
    print(f"MirrorDifferenceNet has {net.n_params} learned numbers")
    x = torch.randn(2, 1, 97, 97)
    g = net(x)
    g_mirror = net(torch.flip(x, dims=(-1,)))
    print("output:", g.tolist())
    print("output on mirrored input:", g_mirror.tolist())
    print("the second is exactly the negative of the first")
