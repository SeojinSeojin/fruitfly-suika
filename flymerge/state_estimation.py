"""Track detector centers across a short, causal settling window."""
import numpy as np
from numba import njit

from .game import HEIGHT, TRANSFER_WIDTH
from .rollout_transfer import reconstruct

TICKS_PER_SECOND = 60.
MATCH_RADIUS = 17.
AMBIGUITY = 3.
LOCAL_TICKS = np.array([-22.,-10.,0.])


@njit(cache=True)
def match(a,b,max_dist,ambiguity):
    """Same-tier mutual nearest neighbors, rejecting ambiguous associations."""
    best_a = np.full(len(a),-1,np.int64)
    best_b = np.full(len(b),-1,np.int64)
    for source,target,result in ((a,b,best_a),(b,a,best_b)):
        for i in range(len(source)):
            first,second,chosen = np.inf,np.inf,-1
            for j in range(len(target)):
                if source[i,4] != target[j,4]:
                    continue
                dx = source[i,0]-target[j,0]
                dy = source[i,1]-target[j,1]
                distance = np.sqrt(dx*dx+dy*dy)
                if distance < first:
                    second,first,chosen = first,distance,j
                elif distance < second:
                    second = distance
            if first <= max_dist and second-first >= ambiguity:
                result[i] = chosen
    for i in range(len(a)):
        j = best_a[i]
        if j >= 0 and best_b[j] != i:
            best_a[i] = -1
    return best_a


def observed_body(fruits):
    arr = np.zeros((len(fruits),5))
    arr[:,:2] = fruits[:,:2]*[TRANSFER_WIDTH,HEIGHT]
    arr[:,4] = fruits[:,2]
    return arr


def track(obs,history,velocity=False):
    """History holds (seconds, normalized Nx3 observations), including the latest frame."""
    state = reconstruct(obs)
    board,count,_,_ = state
    if count==0 or len(history)<2:
        return state,0
    times = (np.array([t for t,_ in history])-history[-1][0])*TICKS_PER_SECOND
    if times[0] > -4 or times[0] < -90 or np.any(np.diff(times)<=0):
        return state,0
    latest = board[:count].copy()
    frames = [observed_body(np.asarray(f,dtype=float).reshape(-1,3)) for _,f in history[:-1]]
    matches = [match(latest,old,MATCH_RADIUS,AMBIGUITY) for old in frames]
    accepted = 0
    for i in range(count):
        ids = [j for j in range(len(frames)) if matches[j][i]>=0]
        if not ids:
            continue
        coords = np.vstack([frames[j][matches[j][i],:2] for j in ids]+[latest[i,:2]])
        center = (np.median(coords,axis=0)+latest[i,:2])*.5
        if velocity and len(ids)>=2:
            t = times[ids+[-1]]
            zero = t-t.mean()
            slope = (zero[:,None]*coords).sum(0)/(zero@zero)
            intercept = coords.mean(0)-slope*t.mean()
            residual = np.linalg.norm(coords-intercept-t[:,None]*slope,axis=1).max()
            speed = np.linalg.norm(slope)
            # Require a coherent trend beyond a few pixels of detector jitter.
            if residual <= 2.5 and .35 <= speed <= 2.5 and speed*(-t[0]) >= 6:
                center = (intercept+latest[i,:2])*.5
                board[i,2:4] = slope
                accepted += 1
        board[i,:2] = center
    return state,accepted


def fruits_from_obs(obs):
    points = obs[:-2].reshape(-1,3)
    points = points[(points[:,0]>=0)&(points[:,1]>=0)].copy()
    points[:,2] = np.clip(np.round(points[:,2]*10),0,10)
    return points


def noisy_frame(env,physical,seed):
    """Apply the same local detector-noise profile with an independent frame RNG."""
    rng = np.random.default_rng(seed)
    frame = physical.copy()
    frame[:,:2] /= [TRANSFER_WIDTH,HEIGHT]
    level = env.vision_noise
    if level:
        frame = frame[rng.random(len(frame)) >= .06*level]
        frame[:,:2] += rng.normal(0,.012*level,(len(frame),2))
        changed = rng.random(len(frame)) < .045*level
        frame[changed,2] = np.clip(frame[changed,2]+rng.choice([-1,1],changed.sum()),0,10)
        large = frame[frame[:,2]>=5]
        if len(large) and rng.random()<.18*level:
            extra = large[rng.integers(len(large))].copy()
            extra[:2] += rng.normal(0,.035*level,2)
            extra[2] = rng.integers(0,3)
            frame = np.vstack((frame,extra))
    np.clip(frame[:,:2],0,1,out=frame[:,:2])
    return frame


def local_history(env,obs,seed,drop):
    """Sample existing physics frames at ticks 58, 70, and 80, before the next action."""
    latest = fruits_from_obs(obs)
    if drop <= 1:
        return [(0.,latest)]
    earlier=[noisy_frame(env,env.frames[j][0],np.random.SeedSequence([seed,drop,1091,k]))
             for k,j in enumerate((19,23))]
    return list(zip(LOCAL_TICKS/TICKS_PER_SECOND,[*earlier,latest]))
