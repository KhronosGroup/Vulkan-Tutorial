import torch
import torch.nn as nn
import onnx
import os

# Create models directory
os.makedirs("models", exist_ok=True)

# 1. LPIPS Model (Structural Roadmap)
class TinyLPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, img1, img2):
        f1 = self.features(img1)
        f2 = self.features(img2)
        # Perceptual distance: mean absolute difference of features
        return torch.mean(torch.abs(f1 - f2), dim=(1, 2, 3), keepdim=True).flatten(1)

# 2. Denoiser U-Net (Structural Roadmap)
class DenoisingUNet(nn.Module):
    def __init__(self, in_channels=10, out_channels=3):
        super().__init__()
        # 10 channels = Noisy RGB (3) + Albedo (3) + Normals (3) + Depth (1)
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(32, out_channels, 1)
        
        # Initialize ALL layers so the network acts as a very subtle spatial
        # smoother for the RGB channels.  enc1 applies a gentle 3x3 weighted
        # average that is close to identity (70% center, 30% neighbours).
        # enc2 is identity so the low-res path adds no extra blur.  dec1 uses
        # 95% from the full-res skip to keep the image sharp, with only 5%
        # contribution from the mildly-smoothed low-res path.
        with torch.no_grad():
            # Gentle 3x3 kernel: 70% center pixel + 30% spread over 8 neighbours
            gentle = torch.ones(3, 3, dtype=torch.float32) * (0.30 / 8.0)
            gentle[1, 1] = 0.70

            # enc1: extract R,G,B from input channels 0,1,2 with gentle blur
            self.enc1[0].weight.fill_(0.0)
            self.enc1[0].bias.fill_(0.0)
            for i in range(out_channels):
                self.enc1[0].weight[i, i] = gentle
            # enc2: identity pass-through (no extra blur at low-res)
            self.enc2[0].weight.fill_(0.0)
            self.enc2[0].bias.fill_(0.0)
            for i in range(out_channels):
                self.enc2[0].weight[i, i, 1, 1] = 1.0
            # up1: upsample first 3 channels (stride-2 transposed conv)
            self.up1.weight.fill_(0.0)
            self.up1.bias.fill_(0.0)
            for i in range(out_channels):
                self.up1.weight[i, i, 0, 0] = 1.0
                self.up1.weight[i, i, 0, 1] = 1.0
                self.up1.weight[i, i, 1, 0] = 1.0
                self.up1.weight[i, i, 1, 1] = 1.0
            # dec1: almost entirely use the full-res skip for sharpness
            # 5% from low-res smoothed path + 95% from enc1 skip (sharper)
            self.dec1[0].weight.fill_(0.0)
            self.dec1[0].bias.fill_(0.0)
            for i in range(out_channels):
                self.dec1[0].weight[i, i, 1, 1] = 0.05       # from up1 (smoothed)
                self.dec1[0].weight[i, 32 + i, 1, 1] = 0.95  # from enc1 skip
            # final: pass through first 3 channels of dec1 output
            self.final.weight.fill_(0.0)
            self.final.bias.fill_(0.0)
            for i in range(out_channels):
                self.final.weight[i, i, 0, 0] = 1.0

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        d1 = self.up1(e2)
        if d1.shape != e1.shape:
            d1 = torch.nn.functional.interpolate(d1, size=e1.shape[2:])
        return self.final(self.dec1(torch.cat([d1, e1], dim=1)))

# 3. Error Predictor (Heatmap)
class ErrorPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )
        # Initialize to act as a simple edge detector
        with torch.no_grad():
            # Laplacian-like kernel for edge detection
            k = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]) / 4.0
            self.net[0].weight.fill_(0.0)
            for i in range(3):
                self.net[0].weight[0, i] = k
            self.net[4].weight.fill_(1.0)
            self.net[4].bias.fill_(-0.5)

    def forward(self, x): return self.net(x)

# 4. Confidence Predictor (Temporal)
class ConfidencePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: Current RGB, History RGB, Current Depth, Previous Depth
        # For simplicity, let's say 8 channels (3+3+1+1)
        self.net = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def export_all():
    print("Generating roadmap models...")
    
    # Helper to export and fix version
    def export_and_fix(model, input_dummy, name, input_names=['input'], output_names=['output']):
        torch.onnx.export(model, input_dummy, name, 
                          input_names=input_names, output_names=output_names,
                          opset_version=9) # Try opset 9 directly
        
        # Manually fix IR version if it still comes out too high
        m = onnx.load(name)
        if m.ir_version > 9:
            m.ir_version = 9
            onnx.save(m, name)
        print(f"✓ {name}")

    # LPIPS
    export_and_fix(TinyLPIPS(), (torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)), 
                   "models/lpips_vgg.onnx", input_names=['input0', 'input1'])

    # Denoiser
    export_and_fix(DenoisingUNet(), torch.randn(1, 10, 256, 256), "models/denoiser.onnx")

    # Error Predictor
    export_and_fix(ErrorPredictor(), torch.randn(1, 3, 256, 256), "models/error_predictor.onnx")

    # Confidence Predictor
    export_and_fix(ConfidencePredictor(), torch.randn(1, 8, 256, 256), "models/confidence.onnx")

if __name__ == "__main__":
    export_all()
