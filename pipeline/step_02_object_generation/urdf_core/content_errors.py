"""Explicit generated-asset rejection, distinct from a failed native runtime."""


class AssetContentError(ValueError):
    def __init__(self, reason, *, stage, link=None, part=None, object_id=None):
        super().__init__(reason)
        self.evidence = {'reason': reason, 'stage': stage, 'link': link, 'part': part}
        if object_id is not None:
            self.evidence['object_id'] = object_id


def require_cooked_geometry(results, hull_count, *, link, part, object_id=None):
    """Only an explicit cook rejection, or a valid empty result, identifies bad content."""
    names = [result if isinstance(result, str) else result.name for result in results]
    content_errors = {'RESULT_ERROR_COOKING_FAILED', 'RESULT_ERROR_UNSUPPORTED_APPROXIMATION'}
    unknown = [name for name in names if name not in content_errors | {'RESULT_VALID'}]
    if not names or unknown:
        raise RuntimeError(f'Native cooking observation unavailable for {part}: {unknown or "no callback"}')
    if any(name in content_errors for name in names) or not hull_count:
        raise AssetContentError(f'Rejected cooked collision representation: {part}: {names}; hulls={hull_count}',
                                stage='collision_cook', link=link, part=part, object_id=object_id)


class AppearanceContentError(ValueError):
    """Optional source appearance cannot be transferred to the published geometry."""
