import math
import unittest

from reference.reference_engine import LFMConfig, ReferenceEngine, apply_rope


class ReferenceEngineResetTests(unittest.TestCase):
    def test_reset_clears_lazily_created_convolution_history(self):
        config = LFMConfig(dim=4, n_layers=1, ssm_state_dim=4)
        engine = ReferenceEngine(config, {})
        engine.ssm_states["conv_0"] = [[1.0, 2.0, 3.0] for _ in range(config.dim)]

        engine.reset()

        self.assertNotIn("conv_0", engine.ssm_states)


class ReferenceRopeTests(unittest.TestCase):
    def test_rope_rotates_each_query_and_kv_head(self):
        q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        k = [2.0, 3.0, 4.0, 5.0]

        apply_rope(q, k, head_dim=4, pos=3, rope_base=100.0)

        for values, heads in ((q, 2), (k, 1)):
            for head in range(heads):
                offset = head * 4
                for pair in range(2):
                    i = offset + pair * 2
                    freq = 1.0 / (100.0 ** ((pair * 2) / 4))
                    angle = 3 * freq
                    c, s = math.cos(angle), math.sin(angle)
                    original = ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                                if values is q else [2.0, 3.0, 4.0, 5.0])
                    x, y = original[i], original[i + 1]
                    self.assertAlmostEqual(values[i], x * c - y * s, places=6)
                    self.assertAlmostEqual(values[i + 1], x * s + y * c, places=6)


if __name__ == "__main__":
    unittest.main()
