#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Tests for `hidef.utils` module."""
from igraph import *
import unittest
from hidef import utils


class TestUtils(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_network_perturb(self):
        g = Graph.GRG(100, 0.2)
        perturbed_g = utils.network_perturb(g)
        self.assertTrue(len(g.es) > len(perturbed_g.es))
