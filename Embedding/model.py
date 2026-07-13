"""
model.py
--------
Implements Algorithm 1 (probabilistic UHKG embedding and pattern-driven
classification, Appendix C.1): each node gets a diagonal-Gaussian
embedding z_v ~ N(mu_v, Sigma_v) (Eq. B.6), trained jointly with a
skip-gram negative-sampling embedding loss (Eq. B.5) and a softmax
pattern-classification loss (Eq. B.7/B.8), combined via the joint loss
of Eq. B.9. Optimization uses a small hand-written Adam optimizer, so
the prototype has no heavy external DL-framework dependency.

For the deterministic ablation variant (HKG-Rec, use_confidence=False):
sigma_v is fixed at (near) zero, so z_v collapses to mu_v (a plain
point embedding), matching Appendix A.4.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class Adam:
    """Minimal per-parameter Adam optimizer (no framework dependency)."""

    def __init__(self, shape, lr=cfg.LEARNING_RATE, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad ** 2)
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class ProbabilisticEmbedding:
    """
    mu, log_sigma2: (n_nodes, dim) trainable center-embedding parameters.
    ctx: (n_nodes, dim) trainable context-embedding parameters (standard
         skip-gram uses separate input/output tables).
    """

    def __init__(self, node_ids, node_types, dim=cfg.EMBED_DIM, use_confidence=True, seed=0):
        rng = np.random.default_rng(seed)
        self.node_ids = list(node_ids)
        self.id2idx = {n: i for i, n in enumerate(self.node_ids)}
        self.types = node_types  # dict node_id -> type string
        self.n = len(self.node_ids)
        self.dim = dim
        self.use_confidence = use_confidence

        self.mu = rng.normal(0, 0.1, size=(self.n, dim))
        # deterministic variant: sigma pinned near 0 (no sampling noise)
        init_log_sigma2 = -6.0 if not use_confidence else -1.0
        self.log_sigma2 = np.full((self.n, dim), init_log_sigma2)
        self.ctx = rng.normal(0, 0.1, size=(self.n, dim))

        self.opt_mu = Adam(self.mu.shape)
        self.opt_ls2 = Adam(self.log_sigma2.shape)
        self.opt_ctx = Adam(self.ctx.shape)

        # by-type index, for type-consistent negative sampling
        self.by_type = {}
        for i, n in enumerate(self.node_ids):
            self.by_type.setdefault(self.types[n], []).append(i)

    def sample_z(self, idx, rng):
        """Eq. B.6 reparameterization: z_v = mu_v + sigma_v * eps."""
        if not self.use_confidence:
            return self.mu[idx]
        sigma = np.exp(0.5 * self.log_sigma2[idx])
        eps = rng.normal(size=sigma.shape)
        return self.mu[idx] + sigma * eps

    def negative_samples(self, node_type, k, exclude_idx, rng):
        pool = self.by_type.get(node_type, list(range(self.n)))
        negs = rng.choice(pool, size=min(k, len(pool)), replace=True)
        return negs

    def train_step(self, pair_batch, label_batch, W, b, alpha=cfg.ALPHA_LOSS,
                    n_neg=5, rng=None):
        """
        One mini-batch SGD/Adam update.
        pair_batch: list of (center_idx, context_idx)
        label_batch: list of (h_idx, sp_label) or None entries for
                     unlabeled service entities.
        W, b: classification head (dim x n_classes), (n_classes,)
        Returns (updated W, b, total_loss) - mu/log_sigma2/ctx updated in place.
        """
        rng = rng or np.random.default_rng()
        grad_mu = np.zeros_like(self.mu)
        grad_ls2 = np.zeros_like(self.log_sigma2)
        grad_ctx = np.zeros_like(self.ctx)
        total_emb_loss = 0.0

        # ---- embedding loss (Eq. B.5, via negative sampling) -----------
        for c_idx, o_idx in pair_batch:
            z_c = self.sample_z(c_idx, rng)
            u_o = self.ctx[o_idx]
            score_pos = sigmoid(np.dot(z_c, u_o))
            total_emb_loss += -np.log(score_pos + 1e-10)
            grad_pos = (score_pos - 1.0)
            grad_ctx[o_idx] += grad_pos * z_c
            grad_z = grad_pos * u_o

            ctype = self.types[self.node_ids[o_idx]]
            negs = self.negative_samples(ctype, n_neg, o_idx, rng)
            for neg_idx in negs:
                u_n = self.ctx[neg_idx]
                score_neg = sigmoid(np.dot(z_c, u_n))
                total_emb_loss += -np.log(1 - score_neg + 1e-10)
                grad_ctx[neg_idx] += score_neg * z_c
                grad_z += score_neg * u_n

            grad_mu[c_idx] += grad_z
            if self.use_confidence:
                sigma = np.exp(0.5 * self.log_sigma2[c_idx])
                eps = (self.sample_z(c_idx, rng) - self.mu[c_idx]) / (sigma + 1e-8)
                grad_ls2[c_idx] += grad_z * eps * 0.5 * sigma

        # ---- classification loss (Eq. B.8) + gradient into embedding ---
        total_cls_loss = 0.0
        grad_W = np.zeros_like(W)
        grad_b = np.zeros_like(b)
        for h_idx, y in label_batch:
            z = self.mu[h_idx]  # classifier reads the mean embedding
            logits = z @ W + b
            logits -= logits.max()
            q = np.exp(logits)
            q /= q.sum()
            total_cls_loss += -np.log(q[y] + 1e-10)
            dlogits = q.copy()
            dlogits[y] -= 1.0
            grad_W += np.outer(z, dlogits)
            grad_b += dlogits
            grad_mu[h_idx] += alpha * (W @ dlogits)

        n_pairs = max(1, len(pair_batch))
        n_labels = max(1, len(label_batch))
        loss = total_emb_loss / n_pairs + alpha * total_cls_loss / n_labels

        self.mu -= self.opt_mu.step(grad_mu / n_pairs)
        self.log_sigma2 -= self.opt_ls2.step(grad_ls2 / n_pairs)
        self.ctx -= self.opt_ctx.step(grad_ctx / n_pairs)
        if label_batch:
            W -= cfg.LEARNING_RATE * grad_W / n_labels
            b -= cfg.LEARNING_RATE * grad_b / n_labels
        return W, b, loss

    def mean_embedding(self, node_id):
        return self.mu[self.id2idx[node_id]]
