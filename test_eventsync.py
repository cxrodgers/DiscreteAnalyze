from __future__ import absolute_import
import numpy as np
import unittest
from . import EventSync

class test1(unittest.TestCase):
    """Tests sign and offset of delay.
    
    These are very basic sanity checks that show that I can pick up
    a delay of a strong signal regardless of signal length or delay sign.
    """
    def test1(self):
        """Equal lengths, zero delay"""
        sig1 = np.array([0, 1, 0])
        sig2 = np.array([0, 1, 0])
        d, p = EventSync.estimate_delay(sig1, sig2)
        self.assertTrue(d == 0)
    
    def test2(self):
        """Unequal lengths, zero delay, sig2 longer"""
        sig1 = np.array([0, 1, 0, 0])
        sig2 = np.array([0, 1, 0])
        d, p = EventSync.estimate_delay(sig1, sig2)
        self.assertTrue(d == 0)

    def test3(self):
        """Unequal lengths, zero delay, sig1 longer"""
        sig1 = np.array([0, 1, 0])
        sig2 = np.array([0, 1, 0, 0])
        d, p = EventSync.estimate_delay(sig1, sig2)
        self.assertTrue(d == 0)

    def test4(self):
        """Equal lengths, delay -1"""
        sig1 = np.array([0, 1, 0])
        sig2 = np.array([0, 0, 1])
        d, p = EventSync.estimate_delay(sig1, sig2)
        self.assertTrue(d == -1)

    def test5(self):
        """Equal lengths, delay 1"""
        sig1 = np.array([0, 0, 1])
        sig2 = np.array([0, 1, 0])
        d, p = EventSync.estimate_delay(sig1, sig2)
        self.assertTrue(d == 1)

    def test6(self):
        """Unequal lengths, delay 1, sig1 longer"""
        sig1 = np.array([0, 0, 0, 1])
        sig2 = np.array([0, 0, 1])
        d, p = EventSync.estimate_delay(sig1, sig2)
        self.assertTrue(d == 1)

    def test7(self):
        """Unequal lengths, delay 1, sig2 longer"""
        sig1 = np.array([0, 0, 1])
        sig2 = np.array([0, 1, 0, 0])
        d, p = EventSync.estimate_delay(sig1, sig2)
        self.assertTrue(d == 1)

    def test8(self):
        """Unequal lengths, delay -1, sig1 longer"""
        sig1 = np.array([1, 0, 0, 0])
        sig2 = np.array([0, 1, ])
        d, p = EventSync.estimate_delay(sig1, sig2)
        self.assertTrue(d == -1)

    def test9(self):
        """Unequal lengths, delay -1, sig2 longer"""
        sig1 = np.array([0, 1, 0])
        sig2 = np.array([0, 0, 1, 0])
        d, p = EventSync.estimate_delay(sig1, sig2)
        self.assertTrue(d == -1)

    def test9(self):
        """Max negative delay"""
        sig1 = np.array([1, 0, 0])
        sig2 = np.array([0, 0, 0, 1])
        d, p = EventSync.estimate_delay(sig1, sig2)
        self.assertTrue(d == -3)
    
    def test9(self):
        """Max positive delay"""
        sig1 = np.array([0, 0, 1])
        sig2 = np.array([1, 0, 0, 0])
        d, p = EventSync.estimate_delay(sig1, sig2)
        self.assertTrue(d == 2)

class testDataset1(unittest.TestCase):
    def test1(self):
        """Easier test case with known alignment
        
        In this case we know that bt[156:] are matched to nt[:] within
        an error term of 100ms.
        """
        nt = np.loadtxt('test/ntimes')
        bt = np.loadtxt('test/btimes')
        
        fit = EventSync.sync(bt, nt, min_acceptable_error=.01,    
            gross_delay_scales_start=None, gross_delay_scales_factor=2.0,
            clock_sync_guess=.9966361, n_scales=4)
        
        self.assertEqual(len(fit), 804)
        self.assertTrue(np.all(np.diff(fit.xi2yi_table) == -156))
    
    def test2(self):
        """Harder test case with imperfect alignment
        
        In this case the `mt` have error bars of about 500ms and also
        there are many missing and spurious entries.
        
        A failure on this test is not a deal breaker since it is probably
        not well-posed. It would be nice if it could fit at least 40 though,
        which is 80% of the values in `mt`.
        """        
        bt = np.loadtxt('test/btimes')
        mt = np.loadtxt('test/mtimes')
        
        fit = EventSync.sync(bt, mt, min_acceptable_error=.5,    
            gross_delay_scales_start=None, gross_delay_scales_factor=2.0,
            clock_sync_guess=.9966361, n_scales=4)
        
        self.assertTrue(len(fit) > 40)
        self.assertTrue(fit.xi2yi[42] == 0)
        self.assertTrue(fit.xi2yi[580] == 51)
    
    

if __name__ == '__main__':
    unittest.main()