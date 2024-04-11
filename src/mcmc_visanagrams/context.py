from typing import Tuple, TYPE_CHECKING, Optional
from collections import UserList

from mcmc_visanagrams.views import VIEW_MAP

ACCEPTED_VIEWS = VIEW_MAP.keys()

if TYPE_CHECKING:
    from mcmc_visanagrams.views.view_base import BaseView


class Context:

    def __init__(self, size: Tuple[int, int, int], prompt: str, magnitude: float, view: str = None):
        self.size: Tuple[int, int, int] = size
        self.prompt: str = prompt
        self.magnitude: float = magnitude

        if view and view not in ACCEPTED_VIEWS:
            raise ValueError(
                f"View {view} is not an accepted view. Accepted views are {ACCEPTED_VIEWS}")
        self.view: Optional['BaseView'] = VIEW_MAP[view]() if view else None

    def __str__(self):
        return (f"Context(size={self.size}, prompt={self.prompt}, magnitude={self.magnitude}, "
                f"view={self.view})")


class ContextList(UserList):

    def __init__(self, init_list=None):
        super().__init__(init_list)

    def append(self, item: Context):
        if not isinstance(item, Context):
            raise ValueError(
                f"Only Context objects can be appended to ContextList. Got {type(item)}")

        super().append(item)

    def collapse(self):
        sizes = [c.size for c in self]
        prompts = [c.prompt for c in self]
        magnitudes = [c.magnitude for c in self]
        views = [c.view for c in self]
        return sizes, prompts, magnitudes, views
