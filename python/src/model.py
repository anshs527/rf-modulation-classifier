"""1D CNN for RadioML 2018.01a modulation classification.

Architecture follows the VGG-ish pattern of the original DeepSig/O'Shea
modulation-classification work: stacked (Conv1d -> ReLU -> MaxPool) blocks
with growing channel counts, then a small fully connected head.
"""
import torch
import torch.nn as nn


class ModulationClassifier(nn.Module):
    """Classify a raw IQ window into one of 24 modulation classes.

    Input:  (batch, 2, 1024) float32 — I and Q as two channels over 1024 timesteps.
    Output: (batch, 24) float32 — raw logits, one per modulation class.

    The output must NOT be passed through softmax: nn.CrossEntropyLoss
    applies log-softmax internally, and softmaxing twice trains badly
    while crashing never.

    Layer plan (each conv preserves length via padding; each pool halves it):

        input                (batch,   2, 1024)
        conv1 + relu + pool  (batch,  64,  512)
        conv2 + relu + pool  (batch,  64,  256)
        conv3 + relu + pool  (batch, 128,  128)
        conv4 + relu + pool  (batch, 128,   64)
        flatten              (batch, 8192)
        fc1 + relu           (batch, 256)
        dropout
        fc2                  (batch, 24)
    """

    NUM_CLASSES = 24
    INPUT_LENGTH = 1024

    def __init__(self):
        super().__init__()
        # Wide first kernel: at ~8 samples per symbol (RadioML's oversampling),
        # kernel_size=7 lets the first layer see most of a symbol at once.
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

        # One pool instance is reusable everywhere: MaxPool1d has no learned
        # parameters, unlike the convs, which each need their own weights.
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(128 * 64, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, self.NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)  
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        # ============================================================
        # EXERCISE 1: forward()
        # See docs/understanding-guide.md § Exercise 1 for the full explanation.
        #
        # What this must do:
        #   - Pass x through the four conv blocks in order; each block is
        #     conv -> relu -> pool (see the layer plan in the class docstring)
        #   - Flatten everything except the batch dimension
        #   - Pass through fc1 -> relu -> dropout -> fc2
        #   - Return the result of fc2 directly (raw logits — NO softmax)
        #
        # What you have available: x, a (batch, 2, 1024) float32 tensor,
        #   and every layer defined in __init__ (self.conv1..conv4,
        #   self.pool, self.relu, self.fc1, self.dropout, self.fc2)
        # What it must produce: a (batch, 24) float32 tensor of logits
        #
        # Hints (read only if stuck):
        #   - What does each layer do to the tensor's shape? Work it out on
        #     paper first, then check yourself against summary() / prints.
        #   - print(x.shape) after every layer is not cheating, it's how
        #     everyone does it.
        #   - The flatten between conv and linear layers is where most
        #     people get the dimensions wrong. torch.flatten(x, start_dim=1)
        #     or x.view/x.reshape — but which dimensions must survive?
        # ============================================================


def summary(model, input_shape=(1, 2, 1024)):
    """Print per-layer output shapes and parameter counts.

    Works by attaching forward hooks and running a dummy tensor through the
    model — so it requires forward() to be implemented. Use it to check your
    mental model of the shape arithmetic against reality.
    """
    rows = []
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            n_params = sum(p.numel() for p in module.parameters())
            rows.append((name, tuple(output.shape), n_params))
        return hook

    for name, module in model.named_modules():
        if name == "":  # skip the top-level module itself
            continue
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    try:
        with torch.no_grad():
            model(torch.randn(*input_shape))
    finally:
        for h in hooks:
            h.remove()

    print(f"{'layer':<12} {'output shape':<22} {'params':>10}")
    print("-" * 46)
    for name, shape, n_params in rows:
        print(f"{name:<12} {str(shape):<22} {n_params:>10,}")
    total = sum(p.numel() for p in model.parameters())
    print("-" * 46)
    print(f"{'total':<35} {total:>10,}")


if __name__ == "__main__":
    summary(ModulationClassifier())
