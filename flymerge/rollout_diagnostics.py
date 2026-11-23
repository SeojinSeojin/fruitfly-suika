"""Separate starting-board error from physics error on matched local forecasts."""
import json

from .afterstate_eval import load_models
from .planner_experiment import environment
from .policy import np
from .replay import json_file
from .rollout_eval import EXACT,ROOT,rng_for
from .rollout_teacher import PENALTY,clean_branch
from .rollout_transfer import NOMINAL,choose,futures,reconstruct


def value(board,count,current,actions,tails,physics):
    return np.array([np.mean([a+b-PENALTY*end for preview,tail in tails
                     for a,b,end,_ in [clean_branch(board,count,current,preview,physics,action,tail)]])
                     for action in actions])


def main():
    variant = json.loads((ROOT/'development.json').read_text())['selected']
    policy,models = load_models()
    rows = []
    for profile in ('clean','joint'):
        for seed in EXACT:
            env=environment(profile,cap=450)
            obs,_=env.reset(seed=seed)
            for drop in range(1,451):
                if drop in (12,90,170,260):
                    action,details=choose(obs,policy,models,seed+drop,rng_for(seed,drop),variant)
                    _,_,_,preview=reconstruct(obs)
                    tails=futures(preview,rng_for(seed,drop),details['tails'])
                    actions=details['actions']
                    nominal=value(env.body,env.n,env.current,actions,tails,NOMINAL)
                    actual=value(env.body,env.n,env.current,actions,tails,env.physics)
                    true_nominal=int(np.lexsort((actions,-nominal))[0])
                    true_physics=int(np.lexsort((actions,-actual))[0])
                    rows.append(dict(seed=seed,profile=profile,drop=drop,actions=actions,
                                     chosen=action,true_board_nominal=actions[true_nominal],
                                     true_board_and_physics=actions[true_physics],
                                     rebuilt_regret=float(actual.max()-actual[actions.index(action)]),
                                     true_board_regret=float(actual.max()-actual[true_nominal])))
                action,_=policy.decide(obs,np.random.default_rng(seed+drop),'robust',trace=False)
                obs,_,terminal,truncated,_=env.step(action)
                if terminal or truncated:
                    break
    summaries={}
    for profile in ('clean','joint'):
        group=[r for r in rows if r['profile']==profile]
        summaries[profile]=dict(boards=len(group),
            changed_using_true_board=sum(x['chosen']!=x['true_board_nominal'] for x in group),
            further_changed_using_true_physics=sum(x['true_board_nominal']!=x['true_board_and_physics'] for x in group),
            mean_regret_rebuilt=float(np.mean([r['rebuilt_regret'] for r in group])),
            mean_regret_true_board=float(np.mean([r['true_board_regret'] for r in group])))
    json_file(ROOT/'diagnostics.json',dict(variant=variant,summary=summaries,rows=rows))


if __name__ == '__main__':
    main()
