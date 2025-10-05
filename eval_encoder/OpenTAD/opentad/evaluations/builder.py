from mmengine.registry import Registry

EVALUATORS = Registry("evaluators")


def build_evaluator(cfg):
    """Build evaluator."""
    return EVALUATORS.build(cfg)


def remove_duplicate_annotations(ants, tol=1e-3):
    # remove duplicate annotations (same category and starting/ending time)
    valid_events = []
    for event in ants:
        s, e, l = event["segment"][0], event["segment"][1], event["label"]
        # here, we add removing the events whose duration is 0, (HACS)
        if e - s <= 0:
            continue
        valid = True
        for p_event in valid_events:
            if (
                (abs(s - p_event["segment"][0]) <= tol)
                and (abs(e - p_event["segment"][1]) <= tol)
                and (l == p_event["label"])
            ):
                valid = False
                break
        if valid:
            valid_events.append(event)
    return valid_events
