Param(
  [ValidateSet('cu121','cu124','cpu')]
  [string]$Cuda = 'cu121'
)

$ErrorActionPreference = 'Stop'

function Run($cmd) {
  Write-Host (">> " + $cmd) -ForegroundColor Cyan
  iex $cmd
}

Write-Host "Setting up PyTorch + PyG for Windows ($Cuda)" -ForegroundColor Green

# 1) Uninstall conflicting packages
Run "pip uninstall -y torch torchvision torchaudio torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv"

# 2) Install CUDA-enabled (or CPU) PyTorch
switch ($Cuda) {
  'cpu'  { $TorchIndex = 'https://download.pytorch.org/whl/cpu' }
  'cu121'{ $TorchIndex = 'https://download.pytorch.org/whl/cu121' }
  'cu124'{ $TorchIndex = 'https://download.pytorch.org/whl/cu124' }
}
Run "pip install --index-url $TorchIndex torch torchvision torchaudio"

# Discover installed torch version (e.g., 2.4.0) for matching PyG wheels
$TorchVer = (& python -c "import torch; print(torch.__version__.split('+')[0])").Trim()
Write-Host "Detected torch version: $TorchVer" -ForegroundColor Yellow

# 3) Install PyG binary wheels for your torch+cuda combo
if ($Cuda -eq 'cpu') {
  $WheelUrl = "https://data.pyg.org/whl/torch-$TorchVer+cpu.html"
} else {
  $WheelUrl = "https://data.pyg.org/whl/torch-$TorchVer+$Cuda.html"
}
Run "pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f $WheelUrl"
Run "pip install torch-geometric"

# 4) Install project Python deps
Run "pip install -r requirements.txt"

# 5) Verify
Run "python scripts/check_env.py"

Write-Host "Setup complete." -ForegroundColor Green

