import torch
import torch.autograd as autograd


def compute_w_dist(fake_data_scores, real_data_scores):
    return real_data_scores.mean() - fake_data_scores.mean()


def compute_gp(critic, fake_data, real_data, *critic_args, **critic_kwargs):
    "Computes gradient penalty according to WGAN-GP paper"
    assert real_data.size() == fake_data.size()

    eps = torch.rand_like(real_data)
    interpolations = eps * real_data + (1 - eps) * fake_data
    preds = critic(interpolations, *critic_args, **critic_kwargs)

    grads = autograd.grad(
        outputs=preds,
        inputs=interpolations,
        grad_outputs=torch.ones_like(preds),
        retain_graph=True, create_graph=True, only_inputs=True
    )[0]

    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()

    return gp
