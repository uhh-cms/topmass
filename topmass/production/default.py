# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.util import maybe_import

from topmass.production.features import features, lb_features
from topmass.production.weights import event_weights

ak = maybe_import("awkward")


@producer(
    uses={features, category_ids, event_weights, lb_features},
    produces={features, category_ids, event_weights, lb_features},
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # features
    events = self[features](events, **kwargs)
    events = self[lb_features](events, **kwargs)

    # category ids
    events = self[category_ids](events, **kwargs)

    # event weights
    events = self[event_weights](events, **kwargs)

    return events


"""
@producer(
    uses={
        features, category_ids, event_weights
    },
    produces={
        features, category_ids, event_weights
    },
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # features
    events = self[features](events, **kwargs)
    
    # category ids
    events = self[category_ids](events, **kwargs)

    # event weights
    events = self[event_weights](events, **kwargs)

    return events
"""
