import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.sparse import issparse


class TuringPatternHunter:
    def __init__(self, adata, bin_size=20, device=None):
        """
        PyTorch-accelerated Turing Pattern Hunter

        Key guarantees:
        - Supports CUDA when available.
        - Also prepares CPU indices for safe fallback to CPU computation.
        """
        self.adata = adata
        self.bin_size = bin_size

        # 1) Auto device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 2) Spatial coords
        if "spatial" not in self.adata.obsm:
            raise ValueError("adata.obsm['spatial'] not found!")

        coords = self.adata.obsm["spatial"]
        self.x_min, self.y_min = coords.min(axis=0)
        self.x_max, self.y_max = coords.max(axis=0)

        # 3) Image size (grid)
        self.img_width = int(np.ceil((self.x_max - self.x_min) / bin_size))
        self.img_height = int(np.ceil((self.y_max - self.y_min) / bin_size))

        print(f"初始化猎人 (PyTorch版):")
        print(f"  - 物理范围: X[{self.x_min:.1f}, {self.x_max:.1f}], Y[{self.y_min:.1f}, {self.y_max:.1f}]")
        print(f"  - Bin Size: {bin_size} (微米/像素)")
        print(f"  - 生成图像尺寸: {self.img_height} x {self.img_width} 像素")
        print(f"  - 计算设备: {self.device}")

        # 4) Precompute grid indices (always compute on CPU first)
        x_idx = ((coords[:, 0] - self.x_min) / self.bin_size).astype(int)
        y_idx = ((coords[:, 1] - self.y_min) / self.bin_size).astype(int)

        x_idx = np.clip(x_idx, 0, self.img_width - 1)
        y_idx = np.clip(y_idx, 0, self.img_height - 1)

        indices_cpu = torch.tensor(np.stack([y_idx, x_idx]), device="cpu")
        flat_cpu = indices_cpu[0] * self.img_width + indices_cpu[1]

        self.indices_cpu = indices_cpu
        self.flat_indices_cpu = flat_cpu

        # Also cache on selected device (cuda/cpu)
        if self.device == "cpu":
            self.indices = self.indices_cpu
            self.flat_indices = self.flat_indices_cpu
        else:
            self.indices = self.indices_cpu.to(self.device, non_blocking=True)
            self.flat_indices = self.flat_indices_cpu.to(self.device, non_blocking=True)

        # Placeholders for results
        self.candidates_u = None
        self.candidates_v = None

    def _get_gene_image_tensor(self, gene_names_or_indices, device=None):
        """
        Convert gene expression to binned image tensor on a target device.

        Parameters
        ----------
        gene_names_or_indices: str/int or list-like
            Gene name(s) or index/indices.
        device: str or torch.device, optional
            Target device ("cuda" or "cpu"). Default: self.device.

        Returns
        -------
        torch.Tensor
            If single gene: (H, W)
            If batch: (B, H, W)
        """
        if device is None:
            device = self.device

        # Support range / list / tuple / ndarray / pd.Index
        is_batch = isinstance(gene_names_or_indices, (list, tuple, np.ndarray, pd.Index, range))

        if not is_batch:
            gene_list = [gene_names_or_indices]
        else:
            gene_list = gene_names_or_indices

        # A) Determine var indices
        if isinstance(gene_list, range) or (len(gene_list) > 0 and isinstance(gene_list[0], (int, np.integer))):
            idxs = gene_list
        else:
            idxs = self.adata.var_names.get_indexer(gene_list)

        # B) Extract expression matrix (N_cells, Batch)
        if self.adata.raw is not None:
            X_data = self.adata.raw.X[:, idxs]
        else:
            X_data = self.adata.X[:, idxs]

        if issparse(X_data):
            X_data = X_data.toarray()

        # C) To tensor on target device
        values = torch.tensor(X_data, dtype=torch.float32, device=device)

        expected_cells = self.indices_cpu.shape[1]

        # Case 1: (N_cells,) -> (N_cells, 1)
        if values.ndim == 1:
            values = values.unsqueeze(1)

        # Case 2: (Batch, N_cells) -> (N_cells, Batch)
        if values.shape[0] != expected_cells and values.shape[1] == expected_cells:
            values = values.T

        # Case 3: still mismatch -> attempt safe fix or crash
        if values.shape[0] != expected_cells:
            if values.shape[0] == 1:
                values = values.repeat(expected_cells, 1)
            else:
                raise RuntimeError(
                    f"Shape Mismatch! Expected {expected_cells} cells, got {values.shape}. "
                    "Check adata.X integrity."
                )

        batch_size = values.shape[1]

        # Pick indices on device
        if str(device) == "cpu":
            flat_indices = self.flat_indices_cpu
        else:
            # Avoid re-copy every time: use cached if device matches self.device, else copy
            if device == self.device:
                flat_indices = self.flat_indices
            else:
                flat_indices = self.flat_indices_cpu.to(device, non_blocking=True)

        # D) Scatter add into flattened image
        img_sum = torch.zeros((batch_size, self.img_height * self.img_width), device=device)
        img_sum.index_add_(1, flat_indices, values.T)

        # E) Count map
        ones = torch.ones(values.shape[0], device=device)
        count_map_flat = torch.zeros(self.img_height * self.img_width, device=device)
        count_map_flat.index_add_(0, flat_indices, ones)

        img_sum = img_sum / (count_map_flat.unsqueeze(0) + 1e-8)

        # F) Reshape & log1p
        imgs = img_sum.view(batch_size, self.img_height, self.img_width)
        imgs = torch.log1p(imgs)

        if not is_batch and imgs.shape[0] == 1:
            return imgs.squeeze(0)

        return imgs

    def _create_gaussian_kernel(self, sigma, truncate=4.0, device=None):
        """
        Create a PyTorch Gaussian kernel on a target device.
        Compatible with older torch versions that don't support meshgrid(indexing=...).
        """
        if device is None:
            device = self.device

        radius = int(truncate * sigma + 0.5)
        k_size = 2 * radius + 1

        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        y = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)

        # meshgrid compatibility
        try:
            xx, yy = torch.meshgrid(x, y, indexing="xy")
        except TypeError:
            xx, yy = torch.meshgrid(x, y)

        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        return kernel.view(1, 1, k_size, k_size), radius

    def screen_geometry(
        self,
        sigma_inner=2,
        sigma_outer=5,
        top_n=50,
        batch_size=100,
        min_batch_size=4,
        quantile_high=0.999,
        quantile_low=0.001,
        verbose=True,
    ):
        """
        L1: Parallel DoG scan with robust fallback (guarantee completion).

        Guarantees:
        - Uses the same scoring definition as original: torch.quantile on dog_flat.
        - If CUDA OOM happens:
            1) reduce batch_size
            2) compute quantile on CPU (exact definition unchanged)
            3) if still OOM, switch full computation to CPU for remaining genes

        Notes:
        - For maximum reproducibility across devices, keep float32 (default).
        """

        if verbose:
            print(f">>> L1: 正在扫描 {self.adata.n_vars} 个基因 (自适应 + 兜底)...")
            print(f"    初始 batch_size={batch_size}, min_batch_size={min_batch_size}")
            print(f"    设备优先: {self.device}")

        results = []
        genes = self.adata.var_names
        n_genes = len(genes)

        # We'll try on preferred device first, but can switch to CPU to guarantee completion
        compute_device = self.device
        cur_bs = int(batch_size)

        # Pre-create kernels for both devices (so switching is cheap)
        kernels = {}
        for dev in set([self.device, "cpu"]):
            k_smooth, pad_smooth = self._create_gaussian_kernel(0.5, device=dev)
            k_in, pad_in = self._create_gaussian_kernel(sigma_inner, device=dev)
            k_out, pad_out = self._create_gaussian_kernel(sigma_outer, device=dev)
            kernels[str(dev)] = (k_smooth, pad_smooth, k_in, pad_in, k_out, pad_out)

        i = 0
        with torch.no_grad():
            while i < n_genes:
                end = min(i + cur_bs, n_genes)
                batch_genes = genes[i:end]

                try:
                    k_smooth, pad_smooth, k_in, pad_in, k_out, pad_out = kernels[str(compute_device)]

                    # 1) get images
                    imgs = self._get_gene_image_tensor(range(i, end), device=compute_device)
                    imgs = imgs.unsqueeze(1)  # (B, 1, H, W)

                    # 2) convolutions
                    imgs_smooth = F.conv2d(imgs, k_smooth, padding=pad_smooth)
                    g_in = F.conv2d(imgs_smooth, k_in, padding=pad_in)
                    g_out = F.conv2d(imgs_smooth, k_out, padding=pad_out)
                    dog = g_in - g_out

                    # 3) scores (quantile)
                    dog_flat = dog.view(dog.shape[0], -1)

                    # Try quantile on current device first (matches original behavior on CUDA).
                    # If OOM occurs here, we catch and compute quantile on CPU instead.
                    try:
                        peak = torch.quantile(dog_flat, quantile_high, dim=1)
                        trough = torch.quantile(dog_flat, quantile_low, dim=1)
                        peak_scores = peak.detach().cpu().numpy()
                        trough_scores = trough.detach().cpu().numpy()
                    except RuntimeError as e:
                        msg = str(e).lower()
                        is_oom = ("out of memory" in msg) or ("cuda out of memory" in msg)
                        if not is_oom:
                            raise
                        # Quantile fallback to CPU (same definition, avoids GPU temp buffers)
                        dog_flat_cpu = dog_flat.detach().cpu()
                        peak_scores = torch.quantile(dog_flat_cpu, quantile_high, dim=1).numpy()
                        trough_scores = torch.quantile(dog_flat_cpu, quantile_low, dim=1).numpy()

                    # 4) record
                    for idx, gene in enumerate(batch_genes):
                        results.append(
                            {
                                "gene": gene,
                                "peak_score": float(peak_scores[idx]),
                                "trough_score": float(trough_scores[idx]),
                            }
                        )

                    # 5) progress
                    i = end

                    # Free intermediates
                    del imgs, imgs_smooth, g_in, g_out, dog, dog_flat

                    if verbose and (i % 1000 == 0 or i == n_genes):
                        print(
                            f"    已处理 {i}/{n_genes}... (device={compute_device}, batch={cur_bs})",
                            end="\r",
                        )

                except RuntimeError as e:
                    msg = str(e).lower()
                    is_oom = ("out of memory" in msg) or ("cuda out of memory" in msg)

                    if not is_oom:
                        raise

                    # CUDA OOM handling
                    if compute_device != "cpu" and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if verbose:
                        print(
                            f"\n[OOM] device={compute_device}, genes {i}~{end}, batch={cur_bs}. 启动兜底策略..."
                        )

                    # Step 1: reduce batch size
                    if cur_bs > min_batch_size:
                        cur_bs = max(min_batch_size, cur_bs // 2)
                        if verbose:
                            print(f"    -> batch_size 降到 {cur_bs}，重试同一段。")
                        continue

                    # Step 2: if still OOM on CUDA at min batch, switch compute device to CPU
                    if compute_device != "cpu":
                        compute_device = "cpu"
                        if verbose:
                            print("    -> 切换到 CPU 完整计算以保证跑完（结果定义不变，速度会变慢）。")
                        continue

                    # Step 3: CPU still OOM (rare unless RAM is insufficient)
                    raise RuntimeError(
                        "CPU 也内存不足，无法继续。建议："
                        "1) 增大 bin_size 进一步降分辨率；"
                        "2) 减少基因数量（先筛高变基因）；"
                        "3) 增加机器内存。"
                    ) from e

        if verbose:
            print("\n筛选完成。")

        df = pd.DataFrame(results)
        self.candidates_u = df.nlargest(top_n, "peak_score")
        self.candidates_v = df.nsmallest(top_n, "trough_score")
        return self.candidates_u, self.candidates_v

    def _get_scale_torch(self, img_tensor):
        """
        Estimate feature scale by autocorrelation decay.
        (kept as original behavior)
        """
        h, w = img_tensor.shape
        crop_size = min(h, w, 200)
        cy, cx = h // 2, w // 2
        img_crop = img_tensor[
            cy - crop_size // 2 : cy + crop_size // 2,
            cx - crop_size // 2 : cx + crop_size // 2,
        ]

        img_crop = img_crop - img_crop.mean()

        H, W = img_crop.shape
        padded = F.pad(img_crop, (0, W, 0, H))

        fft_img = torch.fft.rfft2(padded)
        fft_corr = fft_img * torch.conj(fft_img)
        corr_map = torch.fft.irfft2(fft_corr)

        profile = corr_map[0, : min(H, W) // 2]
        profile = profile / (profile.max() + 1e-9)

        idxs = torch.where(profile < 0.5)[0]
        if len(idxs) > 0:
            return idxs[0].item()
        return len(profile)

    def pair_and_validate(self):
        """
        L2/L3: Pairing and physical validation
        (kept as original behavior)
        """
        print(">>> L2/L3: 配对与物理校验 (GPU加速/CPU兜底与否取决于 self.device)...")
        pairs = []

        if self.candidates_u is None or self.candidates_v is None:
            raise RuntimeError("请先运行 screen_geometry() 生成 candidates_u / candidates_v。")

        unique_genes = list(set(self.candidates_u["gene"]) | set(self.candidates_v["gene"]))
        gene_cache = {}

        print(f"    预计算 {len(unique_genes)} 个候选基因的特征...")
        batch_size = 50

        # Use self.device for this stage; user can set device="cpu" if needed
        for i in range(0, len(unique_genes), batch_size):
            batch_g = unique_genes[i : i + batch_size]
            imgs = self._get_gene_image_tensor(batch_g, device=self.device)

            for j, g in enumerate(batch_g):
                if imgs.ndim == 2:
                    img = imgs
                else:
                    img = imgs[j]

                scale = self._get_scale_torch(img)
                gene_cache[g] = {"img": img, "scale": scale}

        u_genes = self.candidates_u["gene"].values
        v_genes = self.candidates_v["gene"].values

        for u_gene in u_genes:
            cache_u = gene_cache[u_gene]
            img_u = cache_u["img"]
            scale_u = cache_u["scale"]

            for v_gene in v_genes:
                if u_gene == v_gene:
                    continue

                cache_v = gene_cache[v_gene]
                img_v = cache_v["img"]
                scale_v = cache_v["scale"]

                mask = (img_u > 0.1) | (img_v > 0.1)
                if mask.sum() < 50:
                    continue

                val_u = img_u[mask]
                val_v = img_v[mask]

                mean_u = val_u.mean()
                mean_v = val_v.mean()
                num = ((val_u - mean_u) * (val_v - mean_v)).sum()
                den = torch.sqrt(((val_u - mean_u) ** 2).sum() * ((val_v - mean_v) ** 2).sum())
                corr = (num / (den + 1e-8)).item()

                if corr > 0:
                    continue
                if scale_v <= scale_u:
                    continue

                ratio = scale_v / (scale_u + 1e-6)

                pairs.append(
                    {
                        "U_gene": u_gene,
                        "V_gene": v_gene,
                        "correlation": corr,
                        "scale_ratio": ratio,
                        "Turing_Score": ratio * abs(corr),
                    }
                )

        results_df = pd.DataFrame(pairs)
        if results_df.empty:
            return pd.DataFrame(columns=["U_gene", "V_gene", "correlation", "scale_ratio", "Turing_Score"])

        results_df = results_df.sort_values("Turing_Score", ascending=False)
        return results_df
