import torch


class SAM(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        base_optimizer=torch.optim.SGD,
        rho=0.05,
        adaptive=False,
        **kwargs,
    ):
        if rho < 0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for param in group["params"]:
                if param.grad is None:
                    continue
                e_w = torch.pow(param, 2) * param.grad if group["adaptive"] else param.grad
                e_w = e_w * scale.to(param)
                param.add_(e_w)
                self.state[param]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.sub_(self.state[param]["e_w"])

        self.base_optimizer.step()
        self._opt_called = True

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SAM requires closure or explicit first_step/second_step usage.")
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if group["adaptive"]:
                    grad = torch.abs(param) * grad
                norms.append(torch.norm(grad, p=2).to(shared_device))

        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)
