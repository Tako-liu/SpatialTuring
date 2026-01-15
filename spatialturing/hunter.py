import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.sparse import issparse

class TuringPatternHunter:
    def __init__(self, adata, bin_size=20, device=None):
        """
        PyTorch-accelerated Turing Pattern Hunter
        """
        self.adata = adata
        self.bin_size = bin_size
        
        # 1. 自动检测计算设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # 2. 获取物理坐标范围
        if 'spatial' not in self.adata.obsm:
            raise ValueError("adata.obsm['spatial'] not found!")
            
        coords = self.adata.obsm['spatial']
        self.x_min, self.y_min = coords.min(axis=0)
        self.x_max, self.y_max = coords.max(axis=0)
        
        # 3. 计算网格尺寸
        self.img_width = int(np.ceil((self.x_max - self.x_min) / bin_size))
        self.img_height = int(np.ceil((self.y_max - self.y_min) / bin_size))
        
        print(f"初始化猎人 (PyTorch版):")
        print(f"  - 物理范围: X[{self.x_min:.1f}, {self.x_max:.1f}], Y[{self.y_min:.1f}, {self.y_max:.1f}]")
        print(f"  - Bin Size: {bin_size} (微米/像素)")
        print(f"  - 生成图像尺寸: {self.img_height} x {self.img_width} 像素")
        print(f"  - 计算设备: {self.device}")

        # 4. 预计算坐标索引 (N_cells,)
        x_idx = ((coords[:, 0] - self.x_min) / self.bin_size).astype(int)
        y_idx = ((coords[:, 1] - self.y_min) / self.bin_size).astype(int)
        
        # 边界保护
        x_idx = np.clip(x_idx, 0, self.img_width - 1)
        y_idx = np.clip(y_idx, 0, self.img_height - 1)
        
        # 转为 Tensor 并缓存
        self.indices = torch.tensor(np.stack([y_idx, x_idx]), device=self.device)
        self.flat_indices = self.indices[0] * self.img_width + self.indices[1]

    def _get_gene_image_tensor(self, gene_names_or_indices):
        """
        内部辅助：将基因表达量转为 GPU 上的 Tensor 图像
        (已修复维度匹配和 range 支持问题)
        """
        # 判断输入是否为批量 (增加对 range 的支持)
        is_batch = isinstance(gene_names_or_indices, (list, tuple, np.ndarray, pd.Index, range))
        
        if not is_batch:
            gene_list = [gene_names_or_indices]
        else:
            gene_list = gene_names_or_indices

        # A. 获取列索引
        # 如果传入的是 range 或整数列表，直接使用
        if isinstance(gene_list, range) or (len(gene_list) > 0 and isinstance(gene_list[0], (int, np.integer))):
            idxs = gene_list
        else:
            # 如果是基因名，转换为索引
            idxs = self.adata.var_names.get_indexer(gene_list)

        # B. 提取表达量矩阵 (N_cells, Batch)
        # 优先检查 raw
        if self.adata.raw is not None:
             X_data = self.adata.raw.X[:, idxs]
        else:
             X_data = self.adata.X[:, idxs]

        if issparse(X_data):
            X_data = X_data.toarray()
            
        # C. 转为 Tensor
        values = torch.tensor(X_data, dtype=torch.float32, device=self.device) 
        
        # --- 🛡️ 维度防御逻辑 (关键修复) ---
        expected_cells = self.indices.shape[1]
        
        # Case 1: 变成 1D (N_cells,) -> 升维到 (N_cells, 1)
        if values.ndim == 1:
            values = values.unsqueeze(1)
            
        # Case 2: 维度转置 (Batch, N_cells) -> (N_cells, Batch)
        if values.shape[0] != expected_cells and values.shape[1] == expected_cells:
             values = values.T

        # Case 3: 形状依然不对 (Crash保护)
        if values.shape[0] != expected_cells:
             # 如果只有 1 行但需要 N 行 (广播)
             if values.shape[0] == 1:
                 values = values.repeat(expected_cells, 1)
             else:
                 raise RuntimeError(f"Shape Mismatch! Expected {expected_cells} cells, got {values.shape}. Check adata.X integrity.")

        batch_size = values.shape[1]
        
        # D. 栅格化 (Scatter Add)
        img_sum = torch.zeros((batch_size, self.img_height * self.img_width), device=self.device)
        
        # values 现在必须是 (N_cells, Batch)，我们需要它的转置 (Batch, N_cells) 来做 index_add_
        img_sum.index_add_(1, self.flat_indices, values.T)
        
        # E. 计算平均值 (Sum / Count)
        ones = torch.ones(values.shape[0], device=self.device)
        count_map_flat = torch.zeros(self.img_height * self.img_width, device=self.device)
        count_map_flat.index_add_(0, self.flat_indices, ones)
        
        img_sum = img_sum / (count_map_flat.unsqueeze(0) + 1e-8) 
        
        # F. Reshape & Log1p
        imgs = img_sum.view(batch_size, self.img_height, self.img_width)
        imgs = torch.log1p(imgs)
        
        # 如果输入是单个基因，降维返回
        if not is_batch and imgs.shape[0] == 1:
            return imgs.squeeze(0)
            
        return imgs

    def _create_gaussian_kernel(self, sigma, truncate=4.0):
        """创建 PyTorch 高斯卷积核"""
        radius = int(truncate * sigma + 0.5)
        k_size = 2 * radius + 1
        
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=self.device)
        y = torch.arange(-radius, radius + 1, dtype=torch.float32, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        return kernel.view(1, 1, k_size, k_size), radius

    def screen_geometry(self, sigma_inner=2, sigma_outer=5, top_n=50, batch_size=100):
        """
        L1: GPU 并行 DoG 扫描
        """
        print(f">>> L1: 正在扫描 {self.adata.n_vars} 个基因 (GPU加速)...")
        results = []
        genes = self.adata.var_names
        n_genes = len(genes)
        
        # 准备卷积核
        k_smooth, pad_smooth = self._create_gaussian_kernel(0.5)
        k_in, pad_in = self._create_gaussian_kernel(sigma_inner)
        k_out, pad_out = self._create_gaussian_kernel(sigma_outer)

        # 批量处理
        for i in range(0, n_genes, batch_size):
            end = min(i + batch_size, n_genes)
            batch_genes = genes[i:end]
            
            # 1. 获取图片 (使用修复后的函数，支持 range)
            imgs = self._get_gene_image_tensor(range(i, end)) 
            imgs = imgs.unsqueeze(1) # (B, 1, H, W)
            
            # 2. 卷积计算
            imgs_smooth = F.conv2d(imgs, k_smooth, padding=pad_smooth)
            g_in = F.conv2d(imgs_smooth, k_in, padding=pad_in)
            g_out = F.conv2d(imgs_smooth, k_out, padding=pad_out)
            dog = g_in - g_out
            
            # 3. 计算分数
            dog_flat = dog.view(dog.shape[0], -1)
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
        """计算特征尺度 (Autocorrelation decay)"""
        h, w = img_tensor.shape
        crop_size = min(h, w, 200)
        cy, cx = h//2, w//2
        img_crop = img_tensor[cy-crop_size//2 : cy+crop_size//2, cx-crop_size//2 : cx+crop_size//2]
        
        img_crop = img_crop - img_crop.mean()
        
        H, W = img_crop.shape
        padded = F.pad(img_crop, (0, W, 0, H))
        
        fft_img = torch.fft.rfft2(padded)
        fft_corr = fft_img * torch.conj(fft_img)
        corr_map = torch.fft.irfft2(fft_corr)
        
        profile = corr_map[0, :min(H, W)//2]
        profile = profile / (profile.max() + 1e-9)
        
        idxs = torch.where(profile < 0.5)[0]
        if len(idxs) > 0:
            return idxs[0].item()
        return len(profile)

    def pair_and_validate(self):
        """
        L2/L3: 配对与物理校验
        """
        print(">>> L2/L3: 配对与物理校验 (GPU加速)...")
        pairs = []
        
        unique_genes = list(set(self.candidates_u['gene']) | set(self.candidates_v['gene']))
        gene_cache = {} 
        
        print(f"    预计算 {len(unique_genes)} 个候选基因的特征...")
        batch_size = 50
        for i in range(0, len(unique_genes), batch_size):
            batch_g = unique_genes[i : i+batch_size]
            imgs = self._get_gene_image_tensor(batch_g)
            
            for j, g in enumerate(batch_g):
                # 如果 batch_size=1, imgs 可能只有 (H, W)，需要兼容
                if imgs.ndim == 2: img = imgs
                else: img = imgs[j]
                
                scale = self._get_scale_torch(img)
                gene_cache[g] = {'img': img, 'scale': scale}

        u_genes = self.candidates_u['gene'].values
        v_genes = self.candidates_v['gene'].values
        
        for u_gene in u_genes:
            cache_u = gene_cache[u_gene]
            img_u = cache_u['img']
            scale_u = cache_u['scale']
            
            for v_gene in v_genes:
                if u_gene == v_gene: continue
                
                cache_v = gene_cache[v_gene]
                img_v = cache_v['img']
                scale_v = cache_v['scale']
                
                mask = (img_u > 0.1) | (img_v > 0.1)
                if mask.sum() < 50: continue
                
                val_u = img_u[mask]
                val_v = img_v[mask]
                
                mean_u = val_u.mean()
                mean_v = val_v.mean()
                num = ((val_u - mean_u) * (val_v - mean_v)).sum()
                den = torch.sqrt(((val_u - mean_u)**2).sum() * ((val_v - mean_v)**2).sum())
                corr = (num / (den + 1e-8)).item()
                
                if corr > 0: continue 
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