"""Test Sphinx docs.

This file is just for demonstrating how Sphinx works.

DELETE THIS FILE AFTER MERGING.
"""
from __future__ import annotations

from dataclasses import dataclass


def add(a: float, b: float) -> float:
    """Add two numbers.

    :param a: first number
    :param b: second number
    :return: sum of a and b
    """
    return a + b


@dataclass
class Point2D:
    """A 2D point.

    :param x: x-coordinate
    :param y: y-coordinate
    """
    x: float
    """X-coordinate of the point."""
    y: float
    """Y-coordinate of the point."""

    def add(self, other: Point2D) -> Point2D:
        """Add two points together.

        :param other: point to add to this point
        :return: sum of this point and other
        """
        return Point2D(self.x + other.x, self.y + other.y)
