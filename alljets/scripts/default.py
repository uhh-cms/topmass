def dr(j1, j2):
    return j1.delta_r(j2)


def combinationtype(b1, b2, j1, j2, j3, j4, correctcomb):
    import awkward as ak
    b1cor = correctcomb.b[:, 0]
    q1cor = correctcomb.w_children[:, 0, 0]
    q2cor = correctcomb.w_children[:, 0, 1]
    b2cor = correctcomb.b[:, 1]
    q3cor = correctcomb.w_children[:, 1, 0]
    q4cor = correctcomb.w_children[:, 1, 1]
    drmax = 0.4
    drb11, drb22, drq11 = (
        (dr(b1, b1cor) < drmax),
        (dr(b2, b2cor) < drmax),
        (dr(j1, q1cor) < drmax),
    )
    drq22, drq33, drq44 = (
        (dr(j2, q2cor) < drmax),
        (dr(j3, q3cor) < drmax),
        (dr(j4, q4cor) < drmax),
    )
    drq21, drq12 = (dr(j2, q1cor) < drmax), (dr(j1, q2cor) < drmax)
    drq43, drq34 = (dr(j4, q3cor) < drmax), (dr(j3, q4cor) < drmax)
    drb21, drb12, drq31 = (
        (dr(b2, b1cor) < drmax),
        (dr(b1, b2cor) < drmax),
        (dr(j3, q1cor) < drmax),
    )
    drq42, drq13, drq24 = (
        (dr(j4, q2cor) < drmax),
        (dr(j1, q3cor) < drmax),
        (dr(j2, q4cor) < drmax),
    )
    drq41, drq32 = (dr(j4, q1cor) < drmax), (dr(j3, q2cor) < drmax)
    drq23, drq14 = (dr(j2, q3cor) < drmax), (dr(j1, q4cor) < drmax)
    # b1b2: 1234 2134 1243 2143, b2b1: 3412 4312 3421 4321
    drlist = [
        (drb11 & drb22 & drq11 & drq22 & drq33 & drq44),
        (drb11 & drb22 & drq21 & drq12 & drq33 & drq44),
        (drb11 & drb22 & drq11 & drq22 & drq43 & drq34),
        (drb11 & drb22 & drq21 & drq12 & drq43 & drq34),
        (drb21 & drb12 & drq31 & drq42 & drq13 & drq24),
        (drb21 & drb12 & drq41 & drq32 & drq13 & drq24),
        (drb21 & drb12 & drq31 & drq42 & drq23 & drq14),
        (drb21 & drb12 & drq41 & drq32 & drq23 & drq14),
    ]
    # test if all jets are matched
    matched = ak.all(
        [
            ak.any([drb11, drb12], axis=0),
            ak.any([drb22, drb21], axis=0),
            ak.any([drq11, drq12, drq13, drq14], axis=0),
            ak.any([drq21, drq22, drq23, drq24], axis=0),
            ak.any([drq31, drq32, drq33, drq34], axis=0),
            ak.any([drq41, drq42, drq43, drq44], axis=0),
        ],
        axis=0,
    )

    type = matched * 1 + ak.any(drlist, axis=0)
    return type


# function to insert values of one awkward array into another at a list of given indices
def insert_at_index(to_insert, where, indices_to_replace):
    import awkward as ak
    from columnflow.columnar_util import flat_np_view
    full_true = ak.full_like(where, True, dtype=bool)
    mask = full_true & indices_to_replace
    flat = flat_np_view(to_insert)
    cut_orig = ak.num(where[mask])
    cut_replaced = ak.unflatten(flat, cut_orig)
    original = where[~mask]
    combined = ak.concatenate((original, cut_replaced), axis=1)
    return combined


def ambiguous_matching(jets, gen_top, dr):
    """
    For each event, this function checks whether reconstructed jets lie within
    a distance delta R < dr of each generator-level parton originating from a
    tt decay.

    It returns an ak.Array with the following fields:
    - ``b1``: matching mask for the b quark from the t decay
    - ``b2``: matching mask for the b quark from the anti-t decay
    - ``q1``: matching mask for the first quark from the t decay
    - ``q2``: matching mask for the second quark from the t decay
    - ``q3``: matching mask for the first quark from the t decay
    - ``q4``: matching mask for the second quark from the anti-t decay
    """
    from columnflow.util import maybe_import
    ak = maybe_import("awkward")

    matches = ak.zip(
        {
            "b1": gen_top.b[:, 0].delta_r(jets) < dr,
            "b2": gen_top.b[:, 1].delta_r(jets) < dr,
            "q1": gen_top.w_children[:, 0, 0].delta_r(jets) < dr,
            "q3": gen_top.w_children[:, 1, 0].delta_r(jets) < dr,
            "q2": gen_top.w_children[:, 0, 1].delta_r(jets) < dr,
            "q4": gen_top.w_children[:, 1, 1].delta_r(jets) < dr,
        }
    )

    return matches
