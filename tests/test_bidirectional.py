from itertools import permutations

import numpy as np
import pytest

from chronotrace.geometry.bidirectional import (
    GradientDynamics,
    ObservationCell,
    build_prefix_catalogue,
    checkpoint_cell,
    estimated_work_split,
    join_chronology,
    pipeline_roundoff_bound,
)


def affine(seed=0,n=4,steps=3,eta=.08,commuting=False):
    rng=np.random.default_rng(seed)
    hs=[]
    bs=[]
    for _j in range(n):
        z=rng.normal(size=(3,3))
        h=z.T@z
        h*=.7/np.linalg.norm(h,2)
        hs.append(np.zeros_like(h) if commuting else h)
        bs.append(rng.normal(size=3))
    def grad(j,x): return x@hs[j].T-bs[j]
    d=GradientDynamics(tuple(f's{i}' for i in range(n)),np.zeros(3),steps,eta,
       np.array([np.linalg.norm(h,2) for h in hs]),grad,'test known affine',True,1e-13)
    return d,np.array(hs),np.array(bs)


def independent_end(d,hs,bs,word):
    x=d.base.copy()
    for j in word:
        for _ in range(d.steps):
            x=x-d.eta*(hs[j]@x-bs[j])
    return x


@pytest.mark.parametrize('seed',range(12))
@pytest.mark.parametrize('precision',['fp64','fp32','fp16'])
def test_all_small_histories_against_independent_oracle(seed,precision):
    d,hs,bs=affine(seed)
    catalogue=build_prefix_catalogue(d,2)
    orders=list(permutations(range(4)))
    endpoints=np.stack([independent_end(d,hs,bs,w) for w in orders])
    for k in range(24):
        y=endpoints[k].astype(precision.replace('fp','float')).astype(float)
        cell=checkpoint_cell(y,precision)
        result=join_chronology(d,catalogue,cell,join_allowance=1e-10)
        expected={tuple(d.names[i] for i in w) for w,e in zip(orders,endpoints,strict=True)
                  if cell.distance(e)<=2*pipeline_roundoff_bound(d)+1e-10}
        assert set(result.live_orders)==expected
        assert result.joined_candidates+result.unmatched_prefixes_excluded==24
        assert result.unique_order==(next(iter(expected)) if len(expected)==1 else None)
        assert result.inverse_nonconverged_steps==0


@pytest.mark.parametrize('max_iterations',[1,2,3,24])
@pytest.mark.parametrize('seed',range(10))
def test_inverse_enclosure_even_when_solver_exhausts(seed,max_iterations):
    d,hs,bs=affine(seed,steps=8,eta=.2)
    rng=np.random.default_rng(seed+50)
    x=rng.normal(size=(5,3))
    y=np.stack([independent_end(d,hs,bs,(1,)) for _ in range(5)])
    # Forward from a non-base state with an independent implementation.
    y=x.copy()
    for _ in range(d.steps):
        y=y-d.eta*(y@hs[1].T-bs[1])
    noise=rng.normal(size=y.shape)*1e-5
    inv,e,_=d.inverse(1,y+noise,np.linalg.norm(noise,axis=1),max_iterations=max_iterations)
    assert np.all(np.linalg.norm(inv-x,axis=1)<=e+1e-12)


@pytest.mark.parametrize('budget',[0,3,6,12,24,1000])
def test_budget_retains_untested_genuine_ambiguity(budget):
    d,hs,bs=affine(9,commuting=True)
    cat=build_prefix_catalogue(d,2)
    cell=checkpoint_cell(independent_end(d,hs,bs,(0,1,2,3)),'fp64')
    r=join_chronology(d,cat,cell,max_replay_gradients=budget)
    assert len(r.live_orders)==24
    assert r.unique_order is None
    assert r.replay_gradient_evaluations<=budget
    assert not r.inferred_relations


def test_box_resolves_ball_ambiguity():
    # Same diagonal Euclidean noise radius does not imply membership in the box.
    cell=ObservationCell(np.array([-.1,-.001]),np.array([.1,.001]),'custom')
    assert np.linalg.norm(np.array([0,.05]))<cell.radius
    assert cell.distance(np.array([0,.05]))>.04


@pytest.mark.parametrize('precision',['fp16','fp32'])
def test_quantization_cells_contain_unrounded_values(precision):
    rng=np.random.default_rng(442)
    for scale in [0,1e-7,.01,1,100]:
        values=rng.normal(size=100)*scale
        observed=values.astype(precision.replace('fp','float')).astype(float)
        cell=checkpoint_cell(observed,precision)
        assert np.all(values>=cell.lower)
        assert np.all(values<=cell.upper)
    obs=np.array([0.,1.,-1.],dtype=precision.replace('fp','float')).astype(float)
    cell=checkpoint_cell(obs,precision)
    assert cell.lower[0]<0<cell.upper[0]
    assert not cell.lower.flags.writeable


def test_work_accounting():
    d,hs,bs=affine()
    cat=build_prefix_catalogue(d,2)
    assert cat.stage_executions==16
    assert cat.gradient_evaluations==48
    before=d.gradient_evaluations
    r=join_chronology(d,cat,checkpoint_cell(independent_end(d,hs,bs,(0,1,2,3)),'fp64'))
    assert r.inverse_stage_executions==16
    assert r.total_gradient_evaluations==cat.gradient_evaluations+d.gradient_evaluations-before
    assert r.inverse_gradient_evaluations>=r.inverse_stage_executions*d.steps
    assert r.replay_gradient_evaluations==r.replay_stage_executions*d.steps


def test_in_place_gradient_does_not_mutate_state():
    def grad(j,x):
        x[:]=.1
        return x
    d=GradientDynamics(('a','b'),np.zeros(2),3,.1,np.zeros(2),grad,'constant map',True)
    cat=build_prefix_catalogue(d,1)
    np.testing.assert_array_equal(d.base,np.zeros(2))
    x=np.ones((1,2))
    d.grad(0,x)
    np.testing.assert_array_equal(x,np.ones((1,2)))
    cell=checkpoint_cell(np.full(2,-.06),'fp64')
    result=join_chronology(d,cat,cell)
    assert len(result.live_orders)==2


def test_reject_wrong_dynamics_catalogue():
    d,_,_=affine(1)
    other,_,_=affine(2)
    cat=build_prefix_catalogue(d,2)
    with pytest.raises(ValueError):
        join_chronology(other,cat,checkpoint_cell(np.zeros(3),'fp64'))


@pytest.mark.parametrize('depth',[0,4,-1,True,1.5])
def test_bad_depth(depth):
    d,_,_=affine()
    with pytest.raises(ValueError):
        build_prefix_catalogue(d,depth)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.])
def test_bad_allowance(bad):
    d,hs,bs=affine()
    cat=build_prefix_catalogue(d,2)
    with pytest.raises(ValueError):
        join_chronology(d,cat,checkpoint_cell(d.base,'fp64'),join_allowance=bad)


@pytest.mark.parametrize('kw',[
    {'names':('a','a')},{'base':np.array([float('nan')])},{'steps':0},{'steps':True},
    {'eta':-1},{'eta':float('nan')},{'lipschitz':np.array([100.,100.])},
    {'lipschitz':np.array([-1.,1.])},{'regularity_source':''},{'roundoff_allowance':-1}])
def test_bad_dynamics(kw):
    args=dict(names=('a','b'),base=np.zeros(2),steps=3,eta=.1,lipschitz=np.array([1.,1.]),
              gradient=lambda j,x:x,regularity_source='test')
    args.update(kw)
    with pytest.raises(ValueError):
        GradientDynamics(**args)


@pytest.mark.parametrize('precision,values',[
    ('fp8',np.zeros(2)),('fp32',np.array([.1])),('fp16',np.array([65504.])),
    ('fp32',np.array([float('nan')])),('fp64',np.array([]))])
def test_bad_observation(precision,values):
    with pytest.raises(ValueError):
        checkpoint_cell(values,precision)


@pytest.mark.parametrize('n',range(2,11))
def test_split_cost_optimality(n):
    import math
    chosen=estimated_work_split(n,8)
    def cost(k):
        return sum(math.perm(n,r) for r in range(1,k+1))+8*sum(
            math.perm(n,r) for r in range(1,n-k+1)
        )
    assert cost(chosen)==min(cost(k) for k in range(1,n))


def test_outside_model_rejected():
    d,hs,bs=affine()
    cat=build_prefix_catalogue(d,2)
    r=join_chronology(d,cat,checkpoint_cell(np.full(3,100.),'fp64'))
    assert r.status=='inconsistent_assumptions'
    assert r.unique_order is None

@pytest.mark.parametrize('field,value',[('eta',.02),('steps',5),('roundoff_allowance',.1)])
def test_metadata_changes_invalidate_catalogue(field,value):
    d,_,_=affine()
    cat=build_prefix_catalogue(d,2)
    setattr(d,field,value)
    with pytest.raises(ValueError):
        join_chronology(d,cat,checkpoint_cell(d.base,'fp64'))
