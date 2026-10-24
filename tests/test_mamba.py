"""Behavioral, numerical, training, and checkpoint tests; stdlib unittest only."""

import ast
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mamba_numpy import (MambaConfig, MambaLM, AdamW, CharTokenizer,
                         clip_grad_norm, generate)
from mamba_numpy.checkpoint import load_checkpoint, save_checkpoint
from mamba_numpy.data import text_batch, selective_recall_batch
from mamba_numpy.nn import CausalDepthwiseConv1D, Parameter, cross_entropy, sigmoid, softplus
from mamba_numpy.sampling import sample_token
from mamba_numpy.scan import selective_scan, selective_scan_backward, affine_prefix
from mamba_numpy.training import TrainConfig, TaskData, train
from scripts.gradcheck import run_gradcheck, compare


class ScanTests(unittest.TestCase):
    def inputs(self, length=7, dtype="float64"):
        rng = np.random.default_rng(3)
        return {"u": rng.normal(size=(2, length, 3)).astype(dtype),
                "delta": rng.uniform(0.01, 1, (2, length, 3)).astype(dtype),
                "A": -rng.uniform(0.1, 2, (3, 4)).astype(dtype),
                "B": rng.normal(size=(2, length, 4)).astype(dtype),
                "C": rng.normal(size=(2, length, 4)).astype(dtype),
                "D": rng.normal(size=3).astype(dtype)}

    def test_scalar_recurrence_by_hand(self):
        args = {"u": np.array([[[2.], [3.]]]), "delta": np.ones((1, 2, 1)),
                "A": np.array([[np.log(0.5)]]), "B": np.ones((1, 2, 1)),
                "C": np.full((1, 2, 1), 2.), "D": np.array([0.25])}
        y, last, _ = selective_scan(**args)
        np.testing.assert_allclose(y.ravel(), [4.5, 8.75])
        np.testing.assert_allclose(last, [[[4.]]])

    def test_parallel_matches_recurrence(self):
        for length in (1, 2, 3, 8, 17):
            args = self.inputs(length)
            initial = np.ones((2, 3, 4)) * 0.7
            seq = selective_scan(**args, initial_state=initial, retain=True)
            par = selective_scan(**args, initial_state=initial, retain=True, mode="parallel")
            np.testing.assert_allclose(seq[0], par[0], atol=2e-14, rtol=2e-14)
            np.testing.assert_allclose(seq[1], par[1], atol=2e-14, rtol=2e-14)
            np.testing.assert_allclose(seq[2].states, par[2].states, atol=2e-14, rtol=2e-14)

    def test_chunk_continuation_with_nonzero_state(self):
        args = self.inputs(9)
        whole = selective_scan(**args)
        def chunk(start, end):
            return {k: v[:, start:end] if k in ("u", "delta", "B", "C") else v for k, v in args.items()}
        y1, h1, _ = selective_scan(**chunk(0, 4))
        y2, h2, _ = selective_scan(**chunk(4, 9), initial_state=h1)
        np.testing.assert_allclose(np.concatenate((y1, y2), axis=1), whole[0])
        np.testing.assert_allclose(h2, whole[1])

    def test_parallel_backward_matches_sequential(self):
        args = self.inputs()
        seq = selective_scan(**args, retain=True)[2]
        par = selective_scan(**args, retain=True, mode="parallel")[2]
        gy = np.random.default_rng(4).normal(size=args["u"].shape)
        g1, g2 = selective_scan_backward(gy, seq), selective_scan_backward(gy, par)
        for k in g1:
            np.testing.assert_allclose(g1[k], g2[k], atol=2e-14, rtol=2e-14)

    def test_invalid_shapes_and_mode(self):
        args = self.inputs()
        with self.assertRaises(ValueError):
            selective_scan(**dict(args, C=args["C"][:, :-1]))
        with self.assertRaises(ValueError):
            selective_scan(**args, mode="unknown")
        with self.assertRaises(ValueError):
            affine_prefix(np.zeros((2, 0, 3)), np.zeros((2, 0, 3)))


class ModelTests(unittest.TestCase):
    def model(self, **kwargs):
        values = {"vocab_size": 11, "d_model": 8, "n_layers": 2, "d_state": 4,
                  "d_conv": 4, "dtype": "float64", "seed": 2}
        values.update(kwargs)
        return MambaLM(MambaConfig(**values))

    def test_full_step_chunk_and_parallel_agree(self):
        for k in (1, 2, 4, 8):
            for tied in (True, False):
                model = self.model(d_conv=k, tie_embeddings=tied)
                ids = np.random.default_rng(0).integers(0, 11, (3, 13))
                full = model.forward(ids)
                stream, state = model.prefill(ids)
                first, partial = model.prefill(ids[:, :3])
                second, _ = model.prefill(ids[:, 3:], partial)
                parallel = model.forward(ids, scan_mode="parallel")
                np.testing.assert_allclose(full, stream, atol=2e-13, rtol=2e-13)
                np.testing.assert_allclose(full, np.concatenate((first, second), axis=1), atol=2e-13, rtol=2e-13)
                np.testing.assert_allclose(full, parallel, atol=2e-13, rtol=2e-13)
                self.assertEqual(sum(s.nbytes for s in state), model.state_bytes(3))

    def test_streaming_float32(self):
        model = self.model(dtype="float32")
        ids = np.random.default_rng(6).integers(0, 11, (2, 31))
        full, (stream, _) = model.forward(ids), model.prefill(ids)
        self.assertEqual(full.dtype, np.float32)
        np.testing.assert_allclose(full, stream, atol=2e-6, rtol=2e-5)

    def test_causality(self):
        model = self.model()
        ids = np.random.default_rng(9).integers(0, 11, (2, 10))
        changed = ids.copy()
        changed[:, 6:] = (changed[:, 6:] + 1) % 11
        np.testing.assert_array_equal(model.forward(ids)[:, :6], model.forward(changed)[:, :6])

    def test_all_parameter_and_scan_gradients(self):
        report = run_gradcheck(samples=3)
        self.assertTrue(report["passed"], json.dumps(report))

    def test_untied_head_gradient(self):
        model = self.model(n_layers=1, tie_embeddings=False)
        rng = np.random.default_rng(14)
        x, y = rng.integers(0, 11, (2, 4)), rng.integers(0, 11, (2, 4))
        model.loss_and_grad(x, y)
        for name, p in model.named_parameters():
            checks = compare(p.data, p.grad.copy(), lambda: cross_entropy(model.forward(x), y)[0], rng, samples=2)
            self.assertTrue(all(c["passed"] for c in checks), name)

    def test_tied_parameter_is_counted_once(self):
        model = self.model()
        params = list(model.named_parameters())
        self.assertEqual(len({id(p) for _, p in params}), len(params))
        self.assertFalse(any(name.startswith("head.") for name, _ in params))

    def test_inference_does_not_overwrite_training_cache(self):
        model = self.model()
        ids = np.array([[1, 2, 3]])
        y = np.array([[2, 3, 4]])
        expected = model.loss_and_grad(ids, y)
        grads = [p.grad.copy() for p in model.parameters()]
        model.zero_grad()
        loss, g = cross_entropy(model.forward(ids, retain=True), y)
        model.prefill(np.array([[4, 5]]))
        model.forward(ids)
        model.backward(g)
        self.assertEqual(loss, expected)
        for actual, reference in zip(model.parameters(), grads):
            np.testing.assert_array_equal(actual.grad, reference)

    def test_backward_cache_lifecycle(self):
        model = self.model()
        ids = np.array([[1, 2]])
        with self.assertRaises(RuntimeError):
            model.backward(np.zeros((1, 2, 11)))
        logits = model.forward(ids, retain=True)
        with self.assertRaises(RuntimeError):
            model.forward(ids, retain=True)
        model.backward(np.ones_like(logits))
        with self.assertRaises(RuntimeError):
            model.backward(np.ones_like(logits))

    def test_loss_decreases_on_fixed_training_batch(self):
        model = self.model(dtype="float32", n_layers=1)
        x = np.tile(np.array([[1, 2, 3, 4, 5, 6]]), (4, 1))
        y = np.tile(np.array([[2, 3, 4, 5, 6, 7]]), (4, 1))
        initial = cross_entropy(model.forward(x), y)[0]
        opt = AdamW(model.named_parameters(), lr=0.01)
        for _ in range(35):
            model.loss_and_grad(x, y)
            clip_grad_norm(model.parameters())
            opt.step()
        final = cross_entropy(model.forward(x), y)[0]
        self.assertLess(final, initial * 0.35)

    def test_input_validation(self):
        model = self.model()
        for ids in (np.array([[11]]), np.array([[-1]]), np.array([[1.5]]), np.empty((1, 0), dtype=int)):
            with self.assertRaises(ValueError):
                model.forward(ids)
        with self.assertRaises(ValueError):
            model.step(np.array([1]), [])

    def test_state_copy_is_independent(self):
        model = self.model()
        states = model.init_state(1)
        copies = [s.copy() for s in states]
        model.prefill(np.array([[1, 2, 3]]), states)
        self.assertTrue(all(not np.any(s.ssm) for s in copies))
        self.assertTrue(any(np.any(s.ssm) for s in states))


class PrimitiveTests(unittest.TestCase):
    def test_stable_activations_and_cross_entropy(self):
        with np.errstate(over="raise", invalid="raise"):
            s = sigmoid(np.array([-1000., 0., 1000.]))
            p = softplus(np.array([-1000., 0., 1000.]))
            loss, grad = cross_entropy(np.array([[[1000., -1000.]]]), np.array([[0]]))
        np.testing.assert_allclose(s, [0., 0.5, 1.])
        np.testing.assert_allclose(p, [0., np.log(2), 1000.])
        self.assertEqual(loss, 0.)
        np.testing.assert_allclose(grad, 0.)

    def test_masked_loss_and_gradient(self):
        logits = np.zeros((2, 3, 5))
        labels = np.array([[1, -100, 2], [-100, 3, -100]])
        loss, grad = cross_entropy(logits, labels)
        self.assertAlmostEqual(loss, np.log(5))
        np.testing.assert_array_equal(grad[labels == -100], 0.)
        np.testing.assert_allclose(grad.sum(axis=-1), 0., atol=1e-16)
        with self.assertRaises(ValueError):
            cross_entropy(logits, np.full((2, 3), -100))

    def test_convolution_tap_order(self):
        conv = CausalDepthwiseConv1D(1, 3, np.random.default_rng(1), np.float64)
        conv.weight.data[:] = np.array([[10.], [100.], [1000.]])
        x = np.array([[[1.], [2.], [3.], [4.]]])
        np.testing.assert_array_equal(conv.forward(x).ravel(), [1000., 2100., 3210., 4320.])

    def test_adamw_and_no_decay(self):
        p = Parameter(np.array([1.]))
        q = Parameter(np.array([1.]), decay=False)
        opt = AdamW([("p", p), ("q", q)], lr=0.1, weight_decay=0.2)
        p.grad[:] = q.grad[:] = 2.
        opt.step()
        self.assertAlmostEqual(p.data[0], 0.88, places=8)
        self.assertAlmostEqual(q.data[0], 0.9, places=8)

    def test_clipping_and_nonfinite_optimizer_guard(self):
        p, q = Parameter(np.ones(2)), Parameter(np.ones(1))
        p.grad[:] = [3, 4]
        q.grad[:] = 0
        self.assertEqual(clip_grad_norm([p, q], 2.5), 5.)
        np.testing.assert_allclose(p.grad, [1.5, 2.])
        opt = AdamW([("p", p), ("q", q)])
        q.grad[0] = np.nan
        with self.assertRaises(FloatingPointError):
            opt.step()
        np.testing.assert_array_equal(p.data, 1.)
        self.assertEqual(opt.t, 0)


class CheckpointTests(unittest.TestCase):
    def test_roundtrip_and_exact_optimizer_resume(self):
        model = MambaLM(MambaConfig(vocab_size=4, d_model=6, n_layers=1, d_state=3, dtype="float64"))
        opt = AdamW(model.named_parameters(), lr=0.003)
        tokenizer = CharTokenizer.fit("abc")
        rng = np.random.default_rng(18)
        x, y = rng.integers(0, 4, (2, 5)), rng.integers(0, 4, (2, 5))
        model.loss_and_grad(x, y)
        opt.step()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.npz"
            save_checkpoint(path, model, optimizer=opt, tokenizer=tokenizer,
                            metadata={"step": 1}, rng_state=rng.bit_generator.state)
            loaded = load_checkpoint(path, load_optimizer=True)
            np.testing.assert_array_equal(model.forward(x), loaded.model.forward(x))
            self.assertEqual(loaded.tokenizer.to_dict(), tokenizer.to_dict())
            self.assertEqual(loaded.optimizer.t, 1)
            new_rng = np.random.default_rng()
            new_rng.bit_generator.state = loaded.rng_state
            np.testing.assert_array_equal(rng.integers(0, 100, 10), new_rng.integers(0, 100, 10))
            model.loss_and_grad(x, y)
            loaded.model.loss_and_grad(x, y)
            opt.step()
            loaded.optimizer.step()
            for (name, p), (_, q) in zip(model.named_parameters(), loaded.model.named_parameters()):
                np.testing.assert_array_equal(p.data, q.data, err_msg=name)
                np.testing.assert_array_equal(opt.m[name], loaded.optimizer.m[name])

    def test_corrupt_checkpoint_is_rejected(self):
        model = MambaLM(MambaConfig(vocab_size=4, d_model=4, n_layers=1, d_state=2))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.npz"
            save_checkpoint(path, model)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            arrays["parameter/embedding.weight"] = np.zeros((1, 1), dtype=np.float32)
            np.savez(path, **arrays)
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_training_resume_matches_uninterrupted_training(self):
        cfg = MambaConfig(d_model=6, n_layers=1, d_state=3, dtype="float64")
        tc = TrainConfig(task="recall", steps=4, seq_len=12, batch_size=2,
                         eval_batches=1, log_every=4, save_every=2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            whole, _ = train(cfg, tc, root / "whole")
            train(cfg, replace(tc, steps=2), root / "split")
            resumed, _ = train(cfg, tc, root / "split", resume_path=root / "split" / "last.npz")
            for p, q in zip(whole.parameters(), resumed.parameters()):
                np.testing.assert_array_equal(p.data, q.data)


class DataAndSamplingTests(unittest.TestCase):
    def test_unicode_tokenizer(self):
        text = "hello مَمبا 🐍\n"
        tok = CharTokenizer.fit(text)
        self.assertEqual(tok.decode(tok.encode(text)), text)
        self.assertEqual(tok.encode("?")[0], 0)
        self.assertEqual(CharTokenizer.from_dict(tok.to_dict()).to_dict(), tok.to_dict())

    def test_text_shift_and_minimum_corpus(self):
        tokens = np.arange(12)
        x, y = text_batch(tokens, 4, 5, np.random.default_rng(1))
        np.testing.assert_array_equal(x + 1, y)
        with self.assertRaises(ValueError):
            text_batch(np.arange(5), 1, 5, np.random.default_rng())

    def test_recall_targets_match_last_write(self):
        x, y = selective_recall_batch(8, 39, np.random.default_rng(8), n_values=4)
        for b in range(x.shape[0]):
            latest, last_position = None, None
            for t, token in enumerate(x[b]):
                if 1 <= token <= 4:
                    latest, last_position = token, t
                if token == 9:
                    self.assertEqual(y[b, t], latest)
                    self.assertGreaterEqual(t - last_position - 1, 4)
                else:
                    self.assertEqual(y[b, t], -100)
        self.assertGreater(np.sum(y != -100), 0)

    def test_tokenizer_is_fit_only_on_train_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "text.txt"
            path.write_text("a" * 90 + "z" * 10, encoding="utf-8")
            data = TaskData(TrainConfig(seq_len=4, val_fraction=0.1), path=path)
            self.assertNotIn("z", data.tokenizer.stoi)
            np.testing.assert_array_equal(data.val_tokens, 0)

    def test_sampling_filters_and_determinism(self):
        logits = np.array([1., 5., 3.])
        rng = np.random.default_rng(1)
        for _ in range(20):
            self.assertEqual(sample_token(logits, rng, temperature=0), 1)
            self.assertEqual(sample_token(logits, rng, top_k=1), 1)
            self.assertEqual(sample_token(logits, rng, top_p=0.1), 1)
        with self.assertRaises(ValueError):
            sample_token(logits, rng, temperature=-1)

    def test_generation_length_seed_and_zero_tokens(self):
        model = MambaLM(MambaConfig(vocab_size=5, d_model=4, n_layers=1, d_state=2))
        p = np.array([1, 2, 3])
        a = generate(model, p, 9, seed=7)
        b = generate(model, p, 9, seed=7)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), 12)
        np.testing.assert_array_equal(generate(model, p, 0), p)
        with self.assertRaises(ValueError):
            generate(model, [], 1)

    def test_config_validation(self):
        for kwargs in ({"d_model": 0}, {"d_state": -1}, {"dtype": "float16"},
                       {"dt_min": 0}, {"dt_rank": 0}, {"n_layers": True}):
            with self.assertRaises(ValueError):
                MambaConfig(**kwargs)

    def test_numpy_only_runtime_imports(self):
        import sys
        allowed = set(sys.stdlib_module_names) | {"numpy"}
        source = Path(__file__).resolve().parents[1] / "mamba_numpy"
        for path in source.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertIn(alias.name.split(".")[0], allowed, path.name)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    self.assertIn(node.module.split(".")[0], allowed, path.name)


if __name__ == "__main__":
    unittest.main()
