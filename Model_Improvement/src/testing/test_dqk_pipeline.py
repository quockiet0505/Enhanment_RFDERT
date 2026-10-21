"""Fast numerical and output-allocation checks without downloading RF-DETR."""
import tempfile
import unittest
from pathlib import Path

import torch
from src.modules_dqk import DARM, DAFD, UFFR, MFA, compute_tv_loss
from src.common.run_versions import reserve_run, checked_run
from src.common.dqk_checkpoints import export_best


class DQKChecks(unittest.TestCase):
    def test_export_restores_actual_best_epoch_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            state = {"class_embed.weight": torch.randn(33, 8)}
            torch.save({"model": state, "args": {}, "best_total_source": "ema"}, path/"checkpoint_best_total.pth")
            torch.save({"model": state, "epoch": 22}, path/"checkpoint_best_ema.pth")
            result = export_best(path, {"num_classes": 32}, [str(i) for i in range(32)], {"dqk": {}})
            loaded = torch.load(result, weights_only=True)
            self.assertEqual(loaded["epoch"], 22)
            self.assertEqual(loaded["model_config"]["num_classes"], 32)
            self.assertEqual(len(loaded["args"]["class_names"]), 32)
            torch.testing.assert_close(loaded["model"]["class_embed.weight"], state["class_embed.weight"])
            self.assertNotIn("epoch", torch.load(path/"checkpoint_best_total.pth", weights_only=True))

    def test_restoration_starts_as_identity_and_has_task_gradient(self):
        module = DARM()
        x = torch.randn(2, 3, 15, 17)
        y, residual = module(x)
        torch.testing.assert_close(y, x)
        self.assertEqual(compute_tv_loss(residual).item(), 0.)
        y.square().mean().backward()
        self.assertGreater(module.conv3.weight.grad.abs().sum().item(), 0.)

    def test_dafd_identity_prior_and_gate_target(self):
        module = DAFD(8)
        x = torch.randn(2, 8, 15, 17)
        y, gates = module(x)
        torch.testing.assert_close(y, x)
        torch.testing.assert_close(torch.stack(gates), torch.full((3,), .8))
        torch.testing.assert_close(.6*module.centers.sigmoid()+.05, torch.tensor([.1,.35,.6]))
        torch.testing.assert_close(.3*module.widths.sigmoid()+.05, torch.tensor([.12,.18,.25]))

    def test_fft_bin_response_even_and_odd(self):
        for h, w in [(16,16), (15,17), (40,40)]:
            module = UFFR(8)
            dc = torch.ones(1,8,h,w)
            torch.testing.assert_close(module(dc), .95*dc)
            k = 3
            wave = torch.cos(2*torch.pi*k*torch.arange(w)/w)[None,None,None,:].expand(1,8,h,w)
            low = torch.exp(torch.tensor(-.5*((2*k/w)/.6)**2))
            gain = .95*low+1.55*(1-low)
            torch.testing.assert_close(module(wave), gain*wave, rtol=1e-5, atol=2e-6)

    def test_channel_mismatch_is_an_error(self):
        for module in [DAFD(8), UFFR(8)]:
            with self.assertRaises(ValueError):
                module(torch.randn(1, 7, 15, 17))

    def test_frequency_autocast_and_backward(self):
        for module in [DAFD(8), UFFR(8)]:
            x = torch.randn(2,8,15,17, requires_grad=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y = module(x)
                y = y[0] if isinstance(y, tuple) else y
                loss = y.square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(y).all())
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(all(p.grad is not None for p in module.parameters()))

    def test_mfa_shape_and_gradient(self):
        module = MFA(8)
        x = torch.randn(2,8,15,17, requires_grad=True)
        y = module(x)
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in module.parameters()))

    def test_versions_skip_old_paths_and_atomic_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"rfdetr_dqk_enhanced_v1").mkdir()
            sentinel = root/"rfdetr_dqk_enhanced_v1/best.pth"
            sentinel.write_bytes(b"keep")
            claimed = {"rfdetr_dqk_enhanced_v2"}
            def claim(name):
                if name in claimed:
                    return False
                claimed.add(name)
                return True
            name, path = reserve_run(root, "rfdetr_dqk_enhanced", claim)
            self.assertEqual(name, "rfdetr_dqk_enhanced_v3")
            self.assertTrue(path.is_dir())
            self.assertEqual(sentinel.read_bytes(), b"keep")
            self.assertEqual(reserve_run(root, "rfdetr_dqk_enhanced", claim)[0], "rfdetr_dqk_enhanced_v4")

    def test_run_paths_reject_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ["../best", "rfdetr_dqk_enhanced", "rfdetr_dqk_enhanced_v0"]:
                with self.assertRaises(ValueError):
                    checked_run(directory, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
