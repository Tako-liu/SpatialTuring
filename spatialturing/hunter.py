import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.sparse import issparse

class TuringPatternHunter:
    def __init__(self, adata, bin_size=20, device=None):
        """
        PyTorch-accelerated Turing Pattern Hunter (Corrected Version)
        
        修正说明:
        1. 采用 Reflect Padding 保护边缘基因 (如 ATML1)。
        2. Scale 计算完全还原 Scipy 逻辑 (无去均值，使用线性卷积)。
        """
        self.adata = adata
        self.bin_size = bin_size

        # 1. 自动设备检测
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 2. 坐标范围处理
        if "spatial" not in self.adata.obsm:
            raise ValueError("adata.obsm['spatial'] not found!")

        coords = self.adata.obsm["spatial"]
        self.x_min, self.y_min = coords.min(axis=0)
        self.x_max, self.y_max = coords.max(axis=0)

        # 3. 网格尺寸
        self.img_width = int(np.ceil((self.x_max - self.x_min) / bin_size))
        self.img_height = int(np.ceil((self.y_max - self.y_min) / bin_size))

        print(f"初始化猎人 (PyTorch 最终修正版):")
        print(f"  - 物理范围: X[{self.x_min:.1f}, {self.x_max:.1f}], Y[{self.y_min:.1f}, {self.y_max:.1f}]")
        print(f"  - Bin Size: {bin_size}")
        print(f"  - 图像尺寸: {self.img_height} x {self.img_width}")
        print(f"  - 计算设备: {self.device}")

        # 4. 预计算坐标索引
        x_idx = ((coords[:, 0] - self.x_min) / self.bin_size).astype(int)
        y_idx = ((coords[:, 1] - self.y_min) / self.bin_size).astype(int)
        
        # 边界安全截断
        x_idx = np.clip(x_idx, 0, self.img_width - 1)
        y_idx = np.clip(y_idx, 0, self.img_height - 1)

        # 缓存索引 (CPU端)
        self.indices_cpu = torch.tensor(np.stack([y_idx, x_idx]), device="cpu")
        self.flat_indices_cpu = self.indices_cpu[0] * self.img_width + self.indices_cpu[1]
        
        # 缓存索引 (设备端)
        if self.device == "cpu":
            self.flat_indices = self.flat_indices_cpu
        else:
            self.flat_indices = self.flat_indices_cpu.to(self.device, non_blocking=True)

        self.candidates_u = None
        self.candidates_v = None

    def _get_gene_image_tensor(self, gene_names_or_indices, device=None):
        """
        核心：将表达量栅格化为 Tensor 图像
        """
        if device is None: device = self.device
        
        # 判断输入是单个基因还是一批基因
        is_batch = isinstance(gene_names_or_indices, (list, tuple, np.ndarray, pd.Index, range))
        gene_list = gene_names_or_indices if is_batch else [gene_names_or_indices]

        # 1. 获取列索引
        if isinstance(gene_list, range) or (len(gene_list) > 0 and isinstance(gene_list[0], (int, np.integer))):
            idxs = gene_list
        else:
            idxs = self.adata.var_names.get_indexer(gene_list)

        # 2. 提取数据矩阵 (支持 sparse)
        if self.adata.raw is not None:
            X_data = self.adata.raw.X[:, idxs]
        else:
            X_data = self.adata.X[:, idxs]

        if issparse(X_data): X_data = X_data.toarray()
        
        # 3. 转为 Tensor 并移动到设备
        values = torch.tensor(X_data, dtype=torch.float32, device=device)
        
        # 形状调整确保为 (N_cells, Batch)
        if values.ndim == 1: values = values.unsqueeze(1)
        
        expected_cells = self.indices_cpu.shape[1]
        # 如果形状反了 (Batch, N_cells)，转置回来
        if values.shape[0] != expected_cells and values.shape[1] == expected_cells:
            values = values.T 

        batch_size = values.shape[1]
        
        # 4. 栅格化 (Rasterization)
        # 获取对应设备的索引
        flat_indices = self.flat_indices if device == self.device else self.flat_indices_cpu.to(device)

        # 累加表达量 (Sum)
        img_sum = torch.zeros((batch_size, self.img_height * self.img_width), device=device)
        img_sum.index_add_(1, flat_indices, values.T)

        # 计算密度 (Count)
        ones = torch.ones(values.shape[0], device=device)
        count_map_flat = torch.zeros(self.img_height * self.img_width, device=device)
        count_map_flat.index_add_(0, flat_indices, ones)

        # 5. 归一化与 Log
        # Sum / Count = Mean Expression per pixel
        img_sum = img_sum / (count_map_flat.unsqueeze(0) + 1e-8)
        
        # Log1p (配合外部脚本的双重 Log 逻辑)
        imgs = torch.log1p(img_sum).view(batch_size, self.img_height, self.img_width)

        if not is_batch and imgs.shape[0] == 1: return imgs.squeeze(0)
        return imgs

    def _create_gaussian_kernel(self, sigma, truncate=4.0, device=None):
        if device is None: device = self.device
        radius = int(truncate * sigma + 0.5)
        k_size = 2 * radius + 1
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        y = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        
        # 兼容 PyTorch 不同版本的 meshgrid
        try:
            xx, yy = torch.meshgrid(x, y, indexing="xy")
        except TypeError:
            xx, yy = torch.meshgrid(x, y)
            
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, k_size, k_size), radius

    def _apply_conv_reflect(self, img, kernel, padding_size):
        """
        辅助函数：使用 Reflect Padding 进行卷积，保护边缘信号。
        模拟 scipy.ndimage.gaussian_filter(mode='reflect')
        """
        # img shape: (Batch, 1, H, W)
        # Pad order: (Left, Right, Top, Bottom)
        img_padded = F.pad(img, (padding_size, padding_size, padding_size, padding_size), mode='reflect')
        return F.conv2d(img_padded, kernel)

    def screen_geometry(self, sigma_inner=3, sigma_outer=8, top_n=50, batch_size=100):
        print(f">>> L1: 正在扫描 {self.adata.n_vars} 个基因 (GPU/Reflect Padding)...")
        results = []
        genes = self.adata.var_names
        n_genes = len(genes)
        
        # 预创建卷积核
        k_smooth, p_smooth = self._create_gaussian_kernel(0.5)
        k_in, p_in = self._create_gaussian_kernel(sigma_inner)
        k_out, p_out = self._create_gaussian_kernel(sigma_outer)

        with torch.no_grad():
            for i in range(0, n_genes, batch_size):
                end = min(i + batch_size, n_genes)
                batch_genes = genes[i:end]
                
                try:
                    # 1. 获取图像
                    imgs = self._get_gene_image_tensor(range(i, end)).unsqueeze(1)
                    
                    # 2. 卷积 (使用 Reflect Padding)
                    imgs_smooth = self._apply_conv_reflect(imgs, k_smooth, p_smooth)
                    g_in = self._apply_conv_reflect(imgs_smooth, k_in, p_in)
                    g_out = self._apply_conv_reflect(imgs_smooth, k_out, p_out)
                    
                    dog = g_in - g_out
                    
                    # 3. 评分 (99.9% 分位点)
                    dog_flat = dog.view(dog.shape[0], -1)
                    peak = torch.quantile(dog_flat, 0.999, dim=1).cpu().numpy()
                    trough = torch.quantile(dog_flat, 0.001, dim=1).cpu().numpy()
                    
                    for idx, gene in enumerate(batch_genes):
                        results.append({'gene': gene, 'peak_score': peak[idx], 'trough_score': trough[idx]})
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n⚠️ 显存不足 (Batch={batch_size})，自动减半重试...")
                        torch.cuda.empty_cache()
                        # 递归调用，减小 batch size
                        return self.screen_geometry(sigma_inner, sigma_outer, top_n, batch_size // 2)
                    raise e
                
                if i % 1000 == 0 or i + batch_size >= n_genes:
                    print(f"    进度: {end}/{n_genes}...", end="\r")

        print("\n✅ 筛选完成。")
        df = pd.DataFrame(results)
        self.candidates_u = df.nlargest(top_n, "peak_score")
        self.candidates_v = df.nsmallest(top_n, "trough_score")
        return self.candidates_u, self.candidates_v

    def _get_scale_torch(self, img_tensor):
        """
        修正版：使用暴力卷积 (conv2d) 100% 还原 Scipy.signal.correlate2d 的行为。
        关键点：
        1. 不去均值 (保留 DC 分量)。
        2. 使用线性卷积模式 (padding='same' 模拟)。
        """
        h, w = img_tensor.shape
        # 1. 裁剪中心
        crop_size = min(h, w, 200)
        cy, cx = h // 2, w // 2
        img_crop = img_tensor[cy - crop_size//2 : cy + crop_size//2, cx - crop_size//2 : cx + crop_size//2]
        
        # ⚠️ 关键：千万不要 img - img.mean()，否则 Scale 会变极小！
        
        # 2. 准备卷积输入
        # 将 img_crop 既作为 Input 也作为 Weight，实现自相关
        H_crop, W_crop = img_crop.shape
        inp = img_crop.view(1, 1, H_crop, W_crop)
        weight = img_crop.view(1, 1, H_crop, W_crop)
        
        # 3. 计算 Padding (模拟 mode='same')
        # 我们希望输出尺寸 = 输入尺寸
        # Conv2d output = Input + 2*Pad - Kernel + 1
        # Input + 2*Pad - Input + 1 = Input => 2*Pad = Input - 1
        pad_h = (H_crop - 1) // 2
        pad_w = (W_crop - 1) // 2
        
        # 4. 执行卷积
        acf = F.conv2d(inp, weight, padding=(pad_h, pad_w))
        
        # 5. 提取 Profile
        # 中心点在 (H//2, W//2)
        mid_h = H_crop // 2
        mid_w = W_crop // 2
        
        # 取右半部分 Profile
        profile = acf[0, 0, mid_h, mid_w:]
        
        # 6. 计算半衰距离
        if profile.max() == 0: return 0
        profile = profile / (profile.max() + 1e-9)
        
        idxs = torch.where(profile < 0.5)[0]
        if len(idxs) > 0:
            return idxs[0].item()
        return len(profile)

    def pair_and_validate(self):
        print(">>> L2/L3: 配对与物理校验...")
        if self.candidates_u is None: return pd.DataFrame()
        
        unique_genes = list(set(self.candidates_u["gene"]) | set(self.candidates_v["gene"]))
        gene_cache = {}
        batch_size = 50
        
        print(f"    正在预计算 {len(unique_genes)} 个候选基因的特征...")

        # 批量处理以利用 GPU
        for i in range(0, len(unique_genes), batch_size):
            bg = unique_genes[i:i+batch_size]
            imgs = self._get_gene_image_tensor(bg)
            
            for j, g in enumerate(bg):
                # 处理单张图像
                img = imgs[j] if imgs.ndim == 3 else imgs
                # 计算 Scale (GPU)
                scale = self._get_scale_torch(img)
                gene_cache[g] = {"img": img, "scale": scale}

        pairs = []
        u_genes = self.candidates_u["gene"].values
        v_genes = self.candidates_v["gene"].values

        for u_gene in u_genes:
            c_u = gene_cache[u_gene]
            for v_gene in v_genes:
                if u_gene == v_gene: continue
                c_v = gene_cache[v_gene]
                
                # 信号强度过滤
                mask = (c_u["img"] > 0.1) | (c_v["img"] > 0.1)
                if mask.sum() < 20: continue

                val_u, val_v = c_u["img"][mask], c_v["img"][mask]
                
                # 相关性计算 (Pearson Correlation 需要去均值)
                vx = val_u - val_u.mean()
                vy = val_v - val_v.mean()
                num = (vx * vy).sum()
                den = torch.sqrt((vx**2).sum() * (vy**2).sum()) + 1e-8
                corr = (num / den).item()

                if corr > 0: continue # 只找负相关
                if c_v["scale"] <= c_u["scale"]: continue # 物理校验: 抑制剂 > 激活剂
                
                ratio = c_v["scale"] / (c_u["scale"] + 1e-6)
                pairs.append({
                    'U_gene': u_gene, 
                    'V_gene': v_gene, 
                    'correlation': corr, 
                    'scale_ratio': ratio, 
                    'Turing_Score': ratio * abs(corr)
                })

        if not pairs:
            return pd.DataFrame(columns=['U_gene', 'V_gene', 'correlation', 'scale_ratio', 'Turing_Score'])
            
        return pd.DataFrame(pairs).sort_values('Turing_Score', ascending=False)