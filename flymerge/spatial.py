"""Action-relative features shared across the 15 drop columns."""
import torch

from .game import HEIGHT, TRANSFER_RADII, TRANSFER_WIDTH

FEATURE_COUNT = 12


def spatial_features(observation):
    batch = len(observation)
    fruit = observation[:, :-2].reshape(batch, -1, 3)
    current, upcoming = observation[:, -2:].unbind(1)
    radii = observation.new_tensor(TRANSFER_RADII)
    radius = radii[(current * 10).round().long().clamp(0, 10)]
    x = torch.linspace(0, 1, 15, device=observation.device)
    aim = radius[:, None] + (TRANSFER_WIDTH-2*radius[:, None])*x
    center = fruit[..., 0]*TRANSFER_WIDTH
    height = fruit[..., 1]*HEIGHT
    tier = fruit[..., 2]
    valid = center >= 0
    other_radius = radii[(tier*10).round().long().clamp(0, 10)]
    dx = (aim[..., None] - center[:, None, :]).abs()
    rr = radius[:, None, None] + other_radius[:, None, :]
    chord = rr.square() - dx.square()
    contact = height[:, None, :] - chord.clamp_min(0).sqrt()
    contact = contact.masked_fill((chord <= 0) | ~valid[:, None, :], HEIGHT)
    landing, first = contact.min(-1)
    first_tier = tier.gather(1, first).masked_fill(landing >= HEIGHT, -1)
    nearest_same = dx.masked_fill(~(valid & (tier == current[:, None]))[:, None, :], TRANSFER_WIDTH).min(-1).values
    nearest_next = dx.masked_fill(~(valid & (tier == upcoming[:, None]))[:, None, :], TRANSFER_WIDTH).min(-1).values
    pressure = ((dx < 2*rr) & valid[:, None, :]).sum(-1).clamp_max(16)/16
    return torch.stack((x.expand(batch, -1), (x-.5).abs().expand(batch, -1),
                        current[:, None].expand(-1, 15), upcoming[:, None].expand(-1, 15),
                        first_tier, ((first_tier >= 0) & (first_tier == current[:, None])).float(),
                        ((first_tier >= 0) & (first_tier == upcoming[:, None])).float(), landing/HEIGHT,
                        nearest_same/TRANSFER_WIDTH, nearest_next/TRANSFER_WIDTH, pressure,
                        (landing >= HEIGHT).float()), dim=-1)
