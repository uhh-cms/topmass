def dr(j1, j2):
    return j1.delta_r(j2)


def combinationtype(b1, b2, j1, j2, j3, j4, correctcomb):
    import awkward as ak
    b1cor = correctcomb[:, 0, 1]
    q1cor = correctcomb[:, 0, 3]
    q2cor = correctcomb[:, 0, 4]
    b2cor = correctcomb[:, 1, 1]
    q3cor = correctcomb[:, 1, 3]
    q4cor = correctcomb[:, 1, 4]
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
