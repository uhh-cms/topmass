# coding: utf-8

"""
Producers that determine the generator-level particles related to a top quark decay.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")


@producer(
    uses={"GenPart.{genPartIdxMother,pdgId,statusFlags}"},
    produces={"gen_top_decay", "gen_top_decay_last_copy","gen_top_decay_last_isHardProcess"},
)
def gen_top_decay_products_test(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new ragged column "gen_top_decay" with one element per hard top quark. Each element is
    a GenParticleArray with five or more objects in a distinct order: top quark, bottom quark,
    W boson, down-type quark or charged lepton, up-type quark or neutrino, and any additional decay
    produces of the W boson (if any, then most likly photon radiations). Per event, the structure
    will be similar to:

    .. code-block:: python

        [
            # event 1
            [
                # top 1
                [t1, b1, W1, q1/l, q2/n(, additional_w_decay_products)],
                # top 2
                [...],
            ],
            # event 2
            ...
        ]
    """
    # find hard top quarks
    abs_id = abs(events.GenPart.pdgId)
    mother_gen_flags = ["isLastCopy", "fromHardProcess"]
    children_gen_flags = ["isFirstCopy", "fromHardProcess"]
    t = events.GenPart[abs_id == 6]
    t_old = t[t.hasFlags("isHardProcess")]
    t = t[t.hasFlags(*children_gen_flags)]
    t = t[~ak.is_none(t, axis=1)]
    t_old = t_old[~ak.is_none(t_old, axis=1)]

    # distinct top quark children (b's and W's)
    t_children = t.distinctChildrenDeep[t.distinctChildrenDeep.hasFlags(*children_gen_flags)]
    t_children_old = t_old.distinctChildrenDeep[t_old.distinctChildrenDeep.hasFlags("isHardProcess")]

    t = events.GenPart[abs_id == 6]
    t = t[t.hasFlags(*mother_gen_flags)]
    t = t[~ak.is_none(t, axis=1)]

    # get b's
    b = t_children[abs(t_children.pdgId) == 5][:, :, 0]
    b_old = t_children_old[abs(t_children_old.pdgId) == 5][:, :, 0]

    # get W's
    w = t_children[abs(t_children.pdgId) == 24][:, :, 0]
    w_old = t_children_old[abs(t_children_old.pdgId) == 24][:, :, 0]

    # distinct W children
    w_children = w.distinctChildrenDeep[w.distinctChildrenDeep.hasFlags(*children_gen_flags)]
    w_children_old = w_old.distinctChildrenDeep[w_old.distinctChildrenDeep.hasFlags("isHardProcess")]
    

    # reorder the first two W children (leptons or quarks) so that the charged lepton / down-type
    # quark is listed first (they have an odd pdgId)
    w_children_firsttwo = w_children[:, :, :2]
    w_children_firsttwo_old = w_children_old[:, :, :2]

    w_children_firsttwo = w_children_firsttwo[(w_children_firsttwo.pdgId % 2 == 0) * 1]
    w_children_firsttwo_old = w_children_firsttwo_old[(w_children_firsttwo_old.pdgId % 2 == 0) * 1]

    w_children_rest = w_children[:, :, 2:]
    w_children_rest_old = w_children_old[:, :, 2:]

    # last Copy func
    last_copy_t = t.distinctChildren[t.distinctChildren.hasFlags(*mother_gen_flags)]
    t_children2 = ak.where(ak.num(last_copy_t) < 4,t_children,last_copy_t)
    #import IPython; IPython.embed()
    b2 = ak.flatten(t_children2[abs(t_children2.pdgId) == 5],2)

    w2 =  ak.flatten(t_children2[abs(t_children2.pdgId) == 24],2)
    last_copy_w = w2.distinctChildrenDeep[w2.distinctChildrenDeep.hasFlags(*mother_gen_flags)]
    first_copy_w = w2.distinctChildrenDeep[w2.distinctChildrenDeep.hasFlags(*children_gen_flags)]
    w_children2 = ak.where(ak.num(last_copy_w) < 4,first_copy_w,last_copy_w)
    w_children2_firsttwo = w_children2[:, :, :2]
    w_children2_firsttwo = w_children2_firsttwo[(w_children2_firsttwo.pdgId % 2 == 0) * 1]
    w_children2_rest = w_children2[:, :, 2:]
    # concatenate to create the structure to return
    groups = ak.concatenate(
        [
            t[:, :, None],
            b[:, :, None],
            w[:, :, None],
            w_children_firsttwo,
            w_children_rest,
        ],
        axis=2,
    )

    groups_old = ak.concatenate(
        [
            t_old[:, :, None],
            b_old[:, :, None],
            w_old[:, :, None],
            w_children_firsttwo_old,
            w_children_rest_old,
        ],
        axis=2,
    )

    groups2 = ak.concatenate(
        [
            t[:, :, None],
            b2[:, :, None],
            w2[:, :, None],
            w_children2_firsttwo,
            w_children2_rest,
        ],
        axis=2,
    )

    # save the column
    events = set_ak_column(events, "gen_top_decay", groups)
    events = set_ak_column(events, "gen_top_decay_last_copy", groups2)
    events = set_ak_column(events, "gen_top_decay_last_isHardProcess", groups_old)
    return events


@gen_top_decay_products_test.skip
def gen_top_decay_products_skip(self: Producer) -> bool:
    """
    Custom skip function that checks whether the dataset is a MC simulation containing top
    quarks in the first place.
    """
    # never skip when there is not dataset
    if not getattr(self, "dataset_inst", None):
        return False

    return self.dataset_inst.is_data or not self.dataset_inst.has_tag("has_top")
