# Installing Bi-DexterousHands

Bi-DexterousHands depends on NVIDIA Isaac Gym, which is not on PyPI and must
be downloaded from NVIDIA's developer portal. This benchmark is optional; the
paper's main results use ``mixed_smacv2`` and MAMuJoCo.

## Steps

1. Apply for access and download Isaac Gym Preview 4 from
   <https://developer.nvidia.com/isaac-gym>.

2. Extract the archive and install the Python package in your conda env:

   ```bash
   cd isaacgym/python
   pip install -e .
   ```

3. Verify:

   ```bash
   python -c "import isaacgym; print(isaacgym.__version__)"
   ```

4. Install Bi-DexterousHands:

   ```bash
   git clone https://github.com/PKU-MARL/DexterousHands.git
   cd DexterousHands/bi-dexhands
   pip install -e .
   ```

## Hardware requirements

Isaac Gym needs an NVIDIA GPU with driver supporting CUDA 11.0+. Headless
training is supported; on-screen rendering needs an X server.

## Notes for TyMARL

The natural group structure for Bi-DexHands is one group per physical hand:

```yaml
# harl/configs/envs_cfgs/bidexhands.yaml
groups:
  - [0]
  - [1]
```

This makes Bi-DexHands a clean K=2 testbed for TyPPO/TySAC.
