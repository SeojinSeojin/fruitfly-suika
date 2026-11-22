"""Frozen circuit features and trainable actor/critic heads."""
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import torch
from torch import nn

from .game import SLOTS
from .graph import INPUTS, NODES, STEPS

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
FEATURES = SLOTS * 3 + 2


class Policy(nn.Module):
    def __init__(self, kind, graph, sparse=False):
        super().__init__()
        self.kind = kind
        if kind == 'mlp':
            self.embedding = nn.Sequential(nn.Linear(FEATURES, 32), nn.Tanh())
            hidden = 32
        else:
            rng = np.random.default_rng(58011)
            projection = np.zeros((FEATURES, INPUTS), np.float32)
            for i in range(FEATURES):
                projection[i, rng.choice(INPUTS, 4, replace=False)] = rng.choice([-1., 1.], 4)
            projection /= np.maximum(1, np.linalg.norm(projection, axis=0))
            self.register_buffer('projection', torch.from_numpy(projection), persistent=False)
            weights = torch.zeros(NODES, NODES)
            for source, target, count, _ in graph['edges' if kind == 'flywire' else 'rewired']:
                weights[source, target] = count ** .5
            weights *= .92 / weights.sum(0).clamp_min(1.)
            self.register_buffer('weights', weights, persistent=False)
            self.sparse = sparse
            if sparse:
                self.register_buffer('csr', weights.T.contiguous().to_sparse_csr(), persistent=False)
            hidden = NODES
        self.actor = nn.Linear(hidden, 15)
        self.critic = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.actor.weight, gain=.01)
        nn.init.orthogonal_(self.critic.weight)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def circuit(self, observation, trace=False):
        direct = torch.nn.functional.pad(observation @ self.projection, (0, NODES - INPUTS))
        state = torch.zeros_like(direct)
        activity = []
        for _ in range(STEPS):
            recurrence = (self.csr @ state.T).T if self.sparse else state @ self.weights
            state = torch.tanh(direct + recurrence)
            if trace:
                activity.append((recurrence, state))
        return (state, direct, activity) if trace else state

    def enable_spatial_head(self):
        from .spatial import FEATURE_COUNT

        self.spatial_head = nn.Sequential(nn.Linear(FEATURE_COUNT, 32), nn.Tanh(), nn.Linear(32, 1))

    def action_logits(self, features, observation):
        logits = self.actor(features)
        if hasattr(self, 'spatial_head'):
            from .spatial import spatial_features

            logits = logits + self.spatial_head(spatial_features(observation)).squeeze(-1)
        return logits

    def features(self, observation):
        return observation if self.kind == 'mlp' else self.circuit(observation)

    def heads(self, features, observation=None):
        hidden = self.embedding(features) if self.kind == 'mlp' else features
        logits = self.actor(hidden) if observation is None else self.action_logits(hidden, observation)
        return logits, self.critic(hidden).squeeze(-1)

    @torch.no_grad()
    def decide(self, observation, random, strategy="sample", trace=True):
        raw = observation
        observation = torch.as_tensor(observation[None], dtype=torch.float32)
        if trace:
            state, direct, activity = self.circuit(observation, trace=True)
        else:
            state = self.circuit(observation)
        probabilities = self.action_logits(state, observation).softmax(-1)[0].numpy().astype(float)
        if strategy == 'sample':
            action = int(random.choice(15, p=probabilities / probabilities.sum()))
        elif strategy == 'greedy':
            action = int(probabilities.argmax())
        elif strategy == 'robust':
            from .robust_planner import choose

            action, _ = choose(raw, np.log(probabilities.clip(1e-8)), random)
        elif strategy in ('planning', 'hybrid', 'lookahead'):
            from .planning import drop_values

            estimates = drop_values(raw, future=strategy == 'lookahead')
            if strategy != 'planning':
                estimates = estimates + .20 * np.log(probabilities.clip(1e-8))
            action = int(estimates.argmax())
        else:
            raise ValueError(f'Unknown strategy: {strategy}')
        if not trace:
            return action, None
        _, _, reference = self.circuit(torch.zeros_like(observation), trace=True)
        pack = lambda values: np.round(values[0].numpy().astype(float), 5).tolist()
        return action, dict(direct=pack(direct), steps=[
            dict(recurrent=pack(recurrent), activation=pack(activation),
                 reference=pack(zero), delta=pack(activation - zero))
            for (recurrent, activation), (_, zero) in zip(activity, reference)])


def restore(path, graph):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    policy = Policy(checkpoint['kind'], graph)
    if checkpoint.get('spatial_head'):
        policy.enable_spatial_head()
    policy.load_state_dict(checkpoint['state'])
    return policy.eval(), checkpoint
