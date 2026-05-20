import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data.datasets import get_hessian_subset_loader
from src.models.resnet_cifar import get_model


def flatten_tensors(tensors):
    return torch.cat([tensor.contiguous().view(-1) for tensor in tensors])


def unflatten_vector(vector, params):
    tensors = []
    offset = 0
    for param in params:
        numel = param.numel()
        tensors.append(vector[offset : offset + numel].view_as(param))
        offset += numel
    return tensors


def _trainable_params(model):
    return [param for param in model.parameters() if param.requires_grad]


def hvp(model, loss, params, vector):
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
    dot = torch.zeros((), device=loss.device)
    for grad, vec in zip(grads, vector):
        if grad is not None:
            dot = dot + torch.sum(grad * vec)
    products = torch.autograd.grad(dot, params, retain_graph=False, allow_unused=True)
    return [torch.zeros_like(param) if product is None else product for product, param in zip(products, params)]


def _loss_on_batch(model, loss_fn, batch, device):
    inputs, labels = batch
    return loss_fn(model(inputs.to(device, non_blocking=True)), labels.to(device, non_blocking=True))


def _average_hvp_flat(model, loss_fn, data_loader, device, vector_flat, max_batches):
    params = _trainable_params(model)
    vector = unflatten_vector(vector_flat, params)
    hvp_sum = torch.zeros_like(vector_flat)
    batches = 0
    for batch in data_loader:
        if batches >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        loss = _loss_on_batch(model, loss_fn, batch, device)
        product = hvp(model, loss, params, vector)
        hvp_sum = hvp_sum + flatten_tensors([item.detach() for item in product])
        batches += 1
    if batches == 0:
        raise ValueError("No batches available for Hessian estimation.")
    return hvp_sum / batches


def estimate_top_eigenvalue_power_iteration(model, loss_fn, data_loader, device, num_iterations=20, max_batches=4):
    model.eval()
    params = _trainable_params(model)
    vector = torch.randn(sum(param.numel() for param in params), device=device)
    vector = vector / (torch.norm(vector) + 1e-12)
    eigenvalue = torch.zeros((), device=device)
    for _ in range(num_iterations):
        hv_flat = _average_hvp_flat(model, loss_fn, data_loader, device, vector, max_batches)
        eigenvalue = torch.dot(vector, hv_flat)
        vector = hv_flat / (torch.norm(hv_flat) + 1e-12)
    return float(eigenvalue.detach().cpu().item())


def estimate_trace_hutchinson(model, loss_fn, data_loader, device, num_samples=20, max_batches=4):
    model.eval()
    params = _trainable_params(model)
    num_params = sum(param.numel() for param in params)
    estimates = []
    for _ in range(num_samples):
        vector = torch.randint(0, 2, (num_params,), device=device, dtype=torch.float32).mul(2).sub(1)
        hv_flat = _average_hvp_flat(model, loss_fn, data_loader, device, vector, max_batches)
        estimates.append(torch.dot(vector, hv_flat).detach())
    return float(torch.stack(estimates).mean().cpu().item())


def estimate_spectrum_lanczos_or_safe_pr(model, loss_fn, data_loader, device, num_probes=20, max_batches=4):
    model.eval()
    params = _trainable_params(model)
    num_params = sum(param.numel() for param in params)
    values = []
    for _ in range(num_probes):
        vector = torch.randn(num_params, device=device)
        vector = vector / (torch.norm(vector) + 1e-12)
        hv_flat = _average_hvp_flat(model, loss_fn, data_loader, device, vector, max_batches)
        values.append(float(torch.dot(vector, hv_flat).detach().cpu().item()))
    proxy = np.abs(np.asarray(values, dtype=np.float64))
    denom = np.sum(proxy**2)
    pr = float((np.sum(proxy) ** 2) / denom) if denom > 0 else 0.0
    return pr, values


def estimate_topk_lanczos(
    model,
    loss_fn,
    data_loader,
    device,
    top_k=20,
    lanczos_steps=30,
    max_batches=4,
    damping=0.0,
    reorthogonalize=True,
    seed=42,
):
    model.eval()
    params = _trainable_params(model)
    n_params = sum(param.numel() for param in params)
    m = max(1, int(lanczos_steps))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    q = torch.randn(n_params, device=device, generator=generator)
    q = q / (torch.norm(q) + 1e-12)
    q_vectors = []
    alphas = []
    betas = []
    for step in range(m):
        if reorthogonalize and q_vectors:
            for prev_q in q_vectors:
                q = q - torch.dot(prev_q, q) * prev_q
            q = q / (torch.norm(q) + 1e-12)
        q_vectors.append(q.detach().clone())
        z = _average_hvp_flat(model, loss_fn, data_loader, device, q, max_batches)
        if damping:
            z = z + float(damping) * q
        alpha = torch.dot(q, z)
        z = z - alpha * q
        if step > 0:
            z = z - betas[-1] * q_vectors[-2]
        if reorthogonalize:
            for prev_q in q_vectors:
                z = z - torch.dot(prev_q, z) * prev_q
        beta = torch.norm(z)
        alphas.append(alpha.detach())
        if step < m - 1:
            betas.append(beta.detach())
        if beta.item() < 1e-10:
            break
        q = z / beta
    diag = torch.stack(alphas).cpu()
    tri = torch.diag(diag)
    if betas:
        off = torch.stack(betas[: len(alphas) - 1]).cpu()
        tri = tri + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    eigvals = torch.linalg.eigvalsh(tri).flip(0)
    return [float(x) for x in eigvals[: int(top_k)].tolist()]


def compute_spectral_metrics_from_eigenvalues(eigenvalues, trace_estimate, eps=1e-8):
    warnings = []
    eig = np.asarray(eigenvalues or [], dtype=np.float64)
    eig = eig[np.isfinite(eig)]
    eig = np.sort(eig)[::-1]
    positive = eig[eig > eps]
    if len(positive) == 0:
        positive = np.asarray([], dtype=np.float64)
        warnings.append("No positive top-k eigenvalues above eps; spectral metrics set to NaN.")
    elif len(positive) < 5:
        warnings.append(f"Only {len(positive)} positive top-k eigenvalue(s); top-5/top-10 mass may equal 1 by construction.")
    elif len(positive) < 10:
        warnings.append(f"Only {len(positive)} positive top-k eigenvalues; top-10 mass may equal 1 by construction.")
    top_k_sum = float(np.sum(positive)) if len(positive) else math.nan
    lambda_max_topk = float(positive[0]) if len(positive) else math.nan
    denom_topk = np.sum(positive)
    if denom_topk > 0:
        probs = positive / denom_topk
        spectral_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        effective_rank_entropy = float(np.exp(spectral_entropy))
        pr_topk = float((np.sum(positive) ** 2) / np.sum(positive**2))
        top_1 = float(np.sum(positive[:1]) / denom_topk)
        top_5 = float(np.sum(positive[:5]) / denom_topk)
        top_10 = float(np.sum(positive[:10]) / denom_topk)
    else:
        spectral_entropy = effective_rank_entropy = pr_topk = math.nan
        top_1 = top_5 = top_10 = math.nan
    lambda_over_topk_sum = float(lambda_max_topk / top_k_sum) if len(positive) and top_k_sum > 0 else math.nan
    if trace_estimate is None or trace_estimate <= 0:
        warnings.append("trace_estimate <= 0; trace-normalized metrics set to NaN.")
        topk_over_trace = math.nan
    else:
        topk_over_trace = float(top_k_sum / trace_estimate) if math.isfinite(top_k_sum) else math.nan
    return {
        "lambda_max_topk": lambda_max_topk,
        "top_k_sum": top_k_sum,
        "top_1_mass_ratio": top_1,
        "top_5_mass_ratio": top_5,
        "top_10_mass_ratio": top_10,
        "participation_ratio_topk": pr_topk,
        "effective_rank_entropy": effective_rank_entropy,
        "spectral_entropy": spectral_entropy,
        "lambda_max_over_topk_sum": lambda_over_topk_sum,
        "top_k_sum_over_trace": topk_over_trace,
        "num_positive_topk_eigenvalues": int(len(positive)),
        "positive_topk_eigenvalues": [float(x) for x in positive.tolist()],
        "warnings": warnings,
    }


def compute_geometry_for_checkpoint(
    checkpoint,
    dataset="cifar10",
    data_root="data",
    subset_size=1024,
    batch_size=128,
    seed=42,
    num_workers=4,
    device="cuda",
    power_iters=20,
    trace_samples=20,
    pr_probes=20,
    max_batches=4,
    use_lanczos=False,
    top_k=20,
    lanczos_steps=30,
    lanczos_damping=0.0,
    lanczos_reorthogonalize=True,
    model_name=None,
):
    actual_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    loaded = torch.load(checkpoint_path, map_location=actual_device)
    checkpoint_model = loaded.get("model") if isinstance(loaded, dict) else None
    lower_stem = checkpoint_path.stem.lower()
    resolved_model = model_name or checkpoint_model or ("vgg16_bn" if "vgg16_bn" in lower_stem else ("resnet34" if "resnet34" in lower_stem else "resnet18"))
    num_classes = loaded.get("num_classes") if isinstance(loaded, dict) else None
    if num_classes is None:
        num_classes = 100 if str(dataset).lower() == "cifar100" else 10
    model = get_model(resolved_model, num_classes=num_classes).to(actual_device)
    state_dict = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
    model.load_state_dict(state_dict)
    model.eval()
    loader = get_hessian_subset_loader(dataset, data_root, subset_size, batch_size, seed, num_workers)
    loss_fn = nn.CrossEntropyLoss()
    top_eigenvalue = estimate_top_eigenvalue_power_iteration(
        model, loss_fn, loader, actual_device, num_iterations=power_iters, max_batches=max_batches
    )
    trace_estimate = estimate_trace_hutchinson(
        model, loss_fn, loader, actual_device, num_samples=trace_samples, max_batches=max_batches
    )
    participation_ratio_approx, eigen_proxy_values = estimate_spectrum_lanczos_or_safe_pr(
        model, loss_fn, loader, actual_device, num_probes=pr_probes, max_batches=max_batches
    )
    topk_values = []
    if use_lanczos:
        topk_values = estimate_topk_lanczos(
            model,
            loss_fn,
            loader,
            actual_device,
            top_k=top_k,
            lanczos_steps=lanczos_steps,
            max_batches=max_batches,
            damping=lanczos_damping,
            reorthogonalize=lanczos_reorthogonalize,
            seed=seed,
        )
    metric_eigs = topk_values if topk_values else [top_eigenvalue]
    spectral = compute_spectral_metrics_from_eigenvalues(metric_eigs, trace_estimate)
    if not use_lanczos:
        spectral["warnings"].append("Lanczos disabled; top-k spectral metrics are based only on top_eigenvalue and are not valid top-k spectrum metrics.")
    lambda_max_over_trace = math.nan
    if trace_estimate is not None and trace_estimate > 0 and top_eigenvalue is not None:
        lambda_max_over_trace = float(top_eigenvalue / trace_estimate)
    return {
        "top_eigenvalue": top_eigenvalue,
        "trace_estimate": trace_estimate,
        "participation_ratio_approx": participation_ratio_approx,
        "eigen_proxy_values": eigen_proxy_values,
        "raw_topk_eigenvalues": topk_values,
        "top_k_eigenvalues": topk_values,
        "lambda_max_over_trace": lambda_max_over_trace,
        "actual_device": str(actual_device),
        "model": resolved_model,
        **spectral,
    }
