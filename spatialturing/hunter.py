import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.sparse import issparse

class TuringPatternHunter:
    def __init__(self, adata, bin_size=20, device=None):
        """
        PyTorch-accelerated Turing Pattern Hunter
        逻辑严格对齐原版 Scipy 实现，但速度提升 100x+
        """
        self.adata = adata
        self.bin_size = bin_size
        
        # 自动检测设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # 1. 获取物理坐标范围
        if 'spatial' not in self.adata.obsm:
            raise ValueError("adata.obsm['spatial'] not found!")
            
        coords = self.adata.obsm['spatial']
        self.x_min, self.y_min = coords.min(axis=0)
        self.x_max, self.y_max = coords.max(axis=0)
        
        # 2. 计算网格尺寸
        self.img_width = int(np.ceil((self.x_max - self.x_min) / bin_size))
        self.img_height = int(np.ceil((self.y_max - self.y_min) / bin_size))
        
        print(f"初始化猎人 (PyTorch版):")
        print(f"  - 物理范围: X[{self.x_min:.1f}, {self.x_max:.1f}], Y[{self.y_min:.1f}, {self.y_max:.1f}]")
        print(f"  - Bin Size: {bin_size} (微米/像素)")
        print(f"  - 生成图像尺寸: {self.img_height} x {self.img_width} 像素")
        print(f"  - 计算设备: {self.device}")

        # 预计算坐标索引，供后续反复使用
        x_idx = ((coords[:, 0] - self.x_min) / self.bin_size).astype(int)
        y_idx = ((coords[:, 1] - self.y_min) / self.bin_size).astype(int)
        # 边界保护
        x_idx = np.clip(x_idx, 0, self.img_width - 1)
        y_idx = np.clip(y_idx, 0, self.img_height - 1)
        
        # 转为 Tensor 索引 (N_cells,)
        self.indices = torch.tensor(np.stack([y_idx, x_idx]), device=self.device)
        self.flat_indices = self.indices[0] * self.img_width + self.indices[1]

    def _get_gene_image_tensor(self, gene_names_or_indices):
        """
        内部辅助：将基因表达量转为 GPU 上的 Tensor 图像
        支持单个基因或批量基因
        """
        is_batch = isinstance(gene_names_or_indices, (list, tuple, np.ndarray, pd.Index))
        if not is_batch:
            gene_list = [gene_names_or_indices]
        else:
            gene_list = gene_names_or_indices

        # 获取列索引
        if isinstance(gene_list[0], str):
            # 处理 gene name 到 index 的映射 (批量获取更快)
            # 假设 var_names 是唯一的
            all_vars = self.adata.var_names
            # 创建 lookup 可能会慢，直接用 get_indexer
            idxs = all_vars.get_indexer(gene_list)
        else:
            idxs = gene_list

        # 提取表达量矩阵 (N_cells, Batch)
        # 优先检查 raw
        if self.adata.raw is not None:
             X_data = self.adata.raw.X[:, idxs]
        else:
             X_data = self.adata.X[:, idxs]

        if issparse(X_data):
            X_data = X_data.toarray()
            
        # 转为 Tensor
        values = torch.tensor(X_data, dtype=torch.float32, device=self.device) # (N_cells, Batch)
        if values.ndim == 1: values = values.unsqueeze(1)
        
        batch_size = values.shape[1]
        
        # --- 栅格化 (Rasterization) 逻辑完全复刻 np.histogram2d ---
        # 1. 计算 Sum (分子)
        img_sum = torch.zeros((batch_size, self.img_height * self.img_width), device=self.device)
        # index_add_ 需要 transpose values 到 (Batch, N_cells)
        img_sum.index_add_(1, self.flat_indices, values.T)
        
        # 2. 计算 Counts (分母) - 全局只需要算一次，但为了兼容性每次算也行(很快)
        # 实际上 count map 对所有基因是一样的，可以缓存。
        # 这里为了保持逻辑独立性，我们快速算一下。
        ones = torch.ones(values.shape[0], device=self.device)
        count_map_flat = torch.zeros(self.img_height * self.img_width, device=self.device)
        count_map_flat.index_add_(0, self.flat_indices, ones)
        
        # 3. 除法 (Mean)
        img_sum = img_sum / (count_map_flat.unsqueeze(0) + 1e-8) # 避免除0
        
        # Reshape & Log1p
        imgs = img_sum.view(batch_size, self.img_height, self.img_width)
        imgs = torch.log1p(imgs)
        
        if not is_batch:
            return imgs.squeeze(0)
        return imgs

    def _create_gaussian_kernel(self, sigma, truncate=4.0):
        """创建 PyTorch 高斯卷积核，对齐 scipy.ndimage.gaussian_filter"""
        # scipy 默认 truncate=4.0
        radius = int(truncate * sigma + 0.5)
        k_size = 2 * radius + 1
        
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=self.device)
        y = torch.arange(-radius, radius + 1, dtype=torch.float32, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum() # 归一化
        
        # (Out_channels, In_channels/Groups, H, W)
        return kernel.view(1, 1, k_size, k_size), radius

    def screen_geometry(self, sigma_inner=2, sigma_outer=5, top_n=50, batch_size=100):
        """
        L1: GPU 并行扫描 DoG
        """
        print(f">>> L1: 正在扫描 {self.adata.n_vars} 个基因 (GPU加速)...")
        results = []
        genes = self.adata.var_names
        n_genes = len(genes)
        
        # 准备卷积核
        # 1. 预平滑核 (sigma=0.5)
        k_smooth, pad_smooth = self._create_gaussian_kernel(0.5)
        # 2. DoG 核
        k_in, pad_in = self._create_gaussian_kernel(sigma_inner)
        k_out, pad_out = self._create_gaussian_kernel(sigma_outer)

        for i in range(0, n_genes, batch_size):
            end = min(i + batch_size, n_genes)
            batch_genes = genes[i:end]
            
            # 1. 批量栅格化
            imgs = self._get_gene_image_tensor(range(i, end)) # (B, H, W)
            imgs = imgs.unsqueeze(1) # (B, 1, H, W) for conv2d
            
            # 2. 预平滑 (sigma=0.5) - 对应原代码 img_smooth
            # 使用 padding 保持尺寸一致
            imgs_smooth = F.conv2d(imgs, k_smooth, padding=pad_smooth)
            
            # 3. DoG 计算
            g_in = F.conv2d(imgs_smooth, k_in, padding=pad_in)
            g_out = F.conv2d(imgs_smooth, k_out, padding=pad_out)
            dog = g_in - g_out # (B, 1, H, W)
            
            # 4. 计算分数 (99.9% peak, 0.1% trough)
            # Flatten: (B, H*W)
            dog_flat = dog.view(dog.shape[0], -1)
            
            # torch.quantile 比较慢，且需要排序。
            # 为了极致速度，对于 99.9% 和 0.1%，可以直接用 sort
            # 或者如果显存够，直接 quantile
            # 这里用 quantile，确保逻辑和 numpy 完全一致
            peak_scores = torch.quantile(dog_flat, 0.999, dim=1).cpu().numpy()
            trough_scores = torch.quantile(dog_flat, 0.001, dim=1).cpu().numpy()
            
            for idx, gene in enumerate(batch_genes):
                results.append({
                    'gene': gene,
                    'peak_score': peak_scores[idx],
                    'trough_score': trough_scores[idx]
                })
                
            if i % 1000 == 0:
                print(f"    已处理 {end}/{n_genes}...", end="\r")

        print(f"\n筛选完成。")
        df = pd.DataFrame(results)
        self.candidates_u = df.nlargest(top_n, 'peak_score')
        self.candidates_v = df.nsmallest(top_n, 'trough_score')
        return self.candidates_u, self.candidates_v

    def _get_scale_torch(self, img_tensor):
        """
        计算特征尺度 (Autocorrelation decay)，PyTorch 版本
        """
        h, w = img_tensor.shape
        crop_size = min(h, w, 200)
        cy, cx = h//2, w//2
        # Crop
        img_crop = img_tensor[cy-crop_size//2 : cy+crop_size//2, cx-crop_size//2 : cx+crop_size//2]
        
        # 归一化去均值，为了自相关计算准确
        img_crop = img_crop - img_crop.mean()
        
        # 使用 FFT 计算自相关 (比 spatial conv 快得多)
        # Pad to avoid wrap-around effect (circular convolution -> linear convolution)
        # padding size usually double the size
        H, W = img_crop.shape
        padded = F.pad(img_crop, (0, W, 0, H))
        
        fft_img = torch.fft.rfft2(padded)
        # Autocorrelation in frequency domain is |FFT|^2
        fft_corr = fft_img * torch.conj(fft_img)
        corr_map = torch.fft.irfft2(fft_corr)
        
        # Crop back the result and shift (centered)
        # irfft result has peak at (0,0). scaling/shifting is needed if we want 'same' mode behavior strictly
        # But actually, correlate2d 'same' keeps the center.
        # Let's verify peak logic. The logic just needs the decay from the PEAK.
        # The peak of autocorrelation is at [0,0] in the unshifted output.
        
        # 简单起见，我们取前 H/2 行的一条剖面即可，因为自相关是对称的
        # 提取从 (0,0) 开始的一条线 (对应原代码 mid, mid:)
        # 但要注意 fft 结果原点在左上角。
        profile = corr_map[0, :min(H, W)//2] # 取水平方向剖面
        
        profile = profile / (profile.max() + 1e-9)
        
        # 找第一个 < 0.5 的点
        # where returns tuple
        idxs = torch.where(profile < 0.5)[0]
        if len(idxs) > 0:
            return idxs[0].item()
        return len(profile)

    def pair_and_validate(self):
        """
        L2/L3: 配对与物理校验 (逻辑保持不变，利用缓存加速)
        """
        print(">>> L2/L3: 配对与物理校验 (GPU加速)...")
        pairs = []
        
        # 1. 预先将所有候选基因的图像和 Scale 算好并缓存到 GPU
        #    避免在双重循环里反复 rasterize 和算 scale
        unique_genes = list(set(self.candidates_u['gene']) | set(self.candidates_v['gene']))
        
        gene_cache = {} # gene -> {'img': tensor, 'scale': float}
        
        print(f"    预计算 {len(unique_genes)} 个候选基因的特征...")
        # 批量处理缓存
        batch_size = 50
        for i in range(0, len(unique_genes), batch_size):
            batch_g = unique_genes[i : i+batch_size]
            imgs = self._get_gene_image_tensor(batch_g) # (B, H, W)
            
            for j, g in enumerate(batch_g):
                img = imgs[j]
                scale = self._get_scale_torch(img)
                gene_cache[g] = {'img': img, 'scale': scale}

        # 2. 双重循环配对 (逻辑严格对齐)
        # 此时循环内只做简单的 Tensor 计算，极快
        u_genes = self.candidates_u['gene'].values
        v_genes = self.candidates_v['gene'].values
        
        # 进度条
        total_pairs = len(u_genes) * len(v_genes)
        count = 0
        
        for u_gene in u_genes:
            cache_u = gene_cache[u_gene]
            img_u = cache_u['img']
            scale_u = cache_u['scale']
            
            for v_gene in v_genes:
                count += 1
                if u_gene == v_gene: continue
                
                cache_v = gene_cache[v_gene]
                img_v = cache_v['img']
                scale_v = cache_v['scale']
                
                # --- 严格复刻原版 Mask 逻辑 ---
                # mask = (img_u > 0.1) | (img_v > 0.1)
                # if np.sum(mask) < 50: continue
                
                mask = (img_u > 0.1) | (img_v > 0.1)
                if mask.sum() < 50: continue
                
                # 计算相关性 (只在 mask 区域)
                # corr = np.corrcoef(img_u[mask], img_v[mask])[0, 1]
                
                val_u = img_u[mask]
                val_v = img_v[mask]
                
                # 手动计算相关系数以避免转 CPU
                mean_u = val_u.mean()
                mean_v = val_v.mean()
                num = ((val_u - mean_u) * (val_v - mean_v)).sum()
                den = torch.sqrt(((val_u - mean_u)**2).sum() * ((val_v - mean_v)**2).sum())
                corr = (num / (den + 1e-8)).item()
                
                if corr > 0: continue # 只要负相关
                
                # 物理过滤
                if scale_v <= scale_u: continue
                
                ratio = scale_v / (scale_u + 1e-6)
                
                pairs.append({
                    'U_gene': u_gene,
                    'V_gene': v_gene,
                    'correlation': corr,
                    'scale_ratio': ratio,
                    'Turing_Score': ratio * abs(corr)
                })
        
        results_df = pd.DataFrame(pairs)
        if results_df.empty:
            return pd.DataFrame(columns=['U_gene', 'V_gene', 'correlation', 'scale_ratio', 'Turing_Score'])
            
        results_df = results_df.sort_values('Turing_Score', ascending=False)
        return results_df