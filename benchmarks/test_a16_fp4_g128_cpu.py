"""CPU checks for the CSV runner; FLUTE/CUDA is never imported.

Run: python -m unittest discover -s benchmarks -p 'test_a16_fp4_g128_cpu.py' -v
PyTorch-only numerical checks are skipped when PyTorch is unavailable.
"""

import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


_RUNNER_PATH = Path(__file__).with_name('bench_a16_fp4_g128.py')
_SPEC = importlib.util.spec_from_file_location('a16_fp4_g128_runner_test', _RUNNER_PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)

try:
    import torch
except ImportError:
    torch = None


class RunnerCpuTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def make_csv(self, text, name='cases.csv'):
        path = self.directory / name
        path.write_text(text, encoding='utf-8-sig')
        return path

    def call_main(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = runner.main(argv)
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_excel_bom_whitespace_extra_columns_and_quoted_id(self):
        path = self.make_csv(
            ' testcase_id , m , k , n ,note\n'
            '"case, quoted", 3,1536,4096,ignored\n'
            ',,,,\n'
            'second,16,512,256,\n')
        self.assertEqual(runner.read_cases(path), [
            dict(testcase_id='case, quoted', m=3, k=1536, n=4096),
            dict(testcase_id='second', m=16, k=512, n=256),
        ])

    def test_csv_rejects_structural_and_numeric_errors(self):
        cases = [
            ('testcase_id,m,k\nx,1,512\n', 'missing columns'),
            ('testcase_id,m,k,n,n\nx,1,512,256,256\n', 'duplicate'),
            ('testcase_id,m,k,n\nx,1,512,256,overflow\n', 'more values'),
            ('testcase_id,m,k,n\n,1,512,256\n', 'empty testcase_id'),
            ('testcase_id,m,k,n\nx,1.5,512,256\n', 'row 2'),
            ('testcase_id,m,k,n\nx,1,512\n', 'row 2'),
            ('testcase_id,m,k,n\n', 'no cases'),
            ('', 'missing columns'),
        ]
        for text, message in cases:
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, message):
                    runner.read_cases(self.make_csv(text))

    def test_workbook_input_explains_csv_export(self):
        for extension in ('.xlsx', '.xls', '.xlsm', '.XLSX'):
            with self.subTest(extension=extension):
                with self.assertRaisesRegex(ValueError, 'Export Excel as CSV'):
                    runner.read_cases(self.directory / ('cases' + extension))

    def test_valid_smoke_shapes_and_arbitrary_positive_m(self):
        for m in (1, 3, 16, 53, 128, 8192):
            runner.validate_case(dict(m=m, k=1536, n=4096))

    def test_invalid_dimensions_and_unsafe_scale_alignment(self):
        for dimensions in ((0, 512, 256), (1, 0, 256), (1, 512, -256),
                           (1, 511, 256), (1, 512, 128), (1, 128, 256),
                           (1, 256, 256), (1, 384, 256)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(ValueError):
                    runner.validate_case(dict(zip(('m', 'k', 'n'), dimensions)))

    def test_dry_run_validates_without_loading_runtime_or_writing_output(self):
        path = self.make_csv('testcase_id,m,k,n\nx,1,1536,4096\n')
        output = self.directory / 'results' / 'result.csv'
        with mock.patch.object(runner, 'load_runtime', side_effect=AssertionError('GPU used')):
            code, stdout, stderr = self.call_main([
                '--csv', str(path), '--result-csv', str(output), '--dry-run'])
        self.assertEqual(code, 0, stderr)
        self.assertIn('1/1 shapes valid', stdout)
        self.assertIn('NO GPU verification', stdout)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix('.metadata.json').exists())

    def test_dry_run_reports_invalid_shape_and_continues(self):
        path = self.make_csv('testcase_id,m,k,n\nbad,1,128,256\ngood,16,512,256\n')
        code, stdout, stderr = self.call_main(['--csv', str(path), '--dry-run'])
        self.assertEqual(code, 1, stderr)
        self.assertIn('bad:', stdout)
        self.assertIn('INVALID:', stdout)
        self.assertIn('good:', stdout)
        self.assertIn('1/2 shapes valid', stdout)

    def test_cli_rejects_invalid_counts_and_thresholds(self):
        path = self.make_csv('testcase_id,m,k,n\nx,1,512,256\n')
        for option, value in (('--num-tests', '0'), ('--tune-seeds', '0'),
                              ('--warmup', '-1'), ('--check-rows', '-1'),
                              ('--check-cols', '-1'), ('--device', '-1'),
                              ('--flush-mb', '0'), ('--max-diff', 'nan'),
                              ('--max-diff', 'inf'), ('--max-diff', '0'),
                              ('--target-sm', 'sm121')):
            with self.subTest(option=option, value=value):
                code, _, stderr = self.call_main([
                    '--csv', str(path), '--dry-run', option, value])
                self.assertEqual(code, 2)
                self.assertIn('error:', stderr)

    def test_input_cannot_be_overwritten_by_result_or_metadata(self):
        path = self.make_csv('testcase_id,m,k,n\nx,1,512,256\n')
        before = path.read_bytes()
        code, _, stderr = self.call_main([
            '--csv', str(path), '--result-csv', str(path), '--dry-run'])
        self.assertEqual(code, 2)
        self.assertIn('must not be overwritten', stderr)
        self.assertEqual(path.read_bytes(), before)
        metadata = self.make_csv('testcase_id,m,k,n\nx,1,512,256\n', 'run.metadata.json')
        code, _, stderr = self.call_main([
            '--csv', str(metadata), '--result-csv', str(self.directory / 'run.csv'), '--dry-run'])
        self.assertEqual(code, 2)
        self.assertIn('must not be overwritten', stderr)

    def test_selected_indices_cover_ends_without_duplicates(self):
        self.assertEqual(runner.selected_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(runner.selected_indices(10, 1), [0])
        for size in (1, 2, 3, 17, 128, 8192):
            for limit in (0, 1, 2, 3, 32, size, size + 1):
                with self.subTest(size=size, limit=limit):
                    indices = runner.selected_indices(size, limit)
                    expected_count = size if limit == 0 else min(size, limit)
                    self.assertEqual(len(indices), expected_count)
                    self.assertEqual(indices, sorted(set(indices)))
                    self.assertEqual(indices[0], 0)
                    self.assertLess(indices[-1], size)
                    if len(indices) > 1:
                        self.assertEqual(indices[-1], size - 1)

    def test_logical_bytes_include_actual_g128_scales_and_output(self):
        # A=1024; W4=65536; S16=2048; D=512 bytes.
        self.assertEqual(runner.logical_bytes(1, 512, 256), 69120)
        self.assertEqual(runner.logical_bytes(8192, 1536, 65536), 1150812160)

    def test_result_prefix_and_csv_roundtrip_keep_failure_diagnostics(self):
        self.assertEqual(runner.RESULT_FIELDS[:11], [
            'testcase_id', 'm', 'k', 'n', 'passed', 'diff', 'max_abs',
            'elapsed_us', 'tflops', 'gbps', 'error'])
        row = dict.fromkeys(runner.RESULT_FIELDS, '')
        row.update(testcase_id='case, quoted', m=1, k=512, n=256, passed=False,
                   error='failure, with comma\nand newline', group_size=128)
        path = self.directory / 'nested' / 'result.csv'
        runner.write_results(path, [row])
        self.assertTrue(path.read_bytes().startswith(b'\xef\xbb\xbf'))
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, runner.RESULT_FIELDS)
            loaded = list(reader)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]['testcase_id'], row['testcase_id'])
        self.assertEqual(loaded[0]['error'], row['error'])
        self.assertEqual(loaded[0]['passed'], 'False')
        self.assertEqual(loaded[0]['elapsed_us'], '')

    def test_invalid_case_returns_failure_without_tensor_operations(self):
        case = dict(testcase_id='invalid', m=-1, k=512, n=256)
        result = runner.run_case(case, SimpleNamespace(a_dtype='bf16'),
                                 SimpleNamespace(torch=object(), flute=object()))
        self.assertFalse(result['passed'])
        self.assertIn('positive', result['error'])
        self.assertEqual(result['elapsed_us'], '')
        self.assertEqual(result['group_size'], 128)

    def test_graph_timer_captures_flush_before_external_events_and_replays(self):
        history = []
        event_options = []
        elapsed_ms = iter((.002, .006, .004))
        captured_output = object()

        class Event:
            def __init__(self, **kwargs):
                event_options.append(kwargs)
                self.name = 'start' if len(event_options) == 1 else 'end'

            def record(self):
                history.append(self.name + '.record')

            def synchronize(self):
                history.append(self.name + '.synchronize')

            def elapsed_time(self, other):
                self_test.assertEqual((self.name, other.name), ('start', 'end'))
                return next(elapsed_ms)

        class Graph:
            def replay(self):
                history.append('replay')

        @contextlib.contextmanager
        def capture(graph):
            history.append('capture_enter')
            yield
            history.append('capture_exit')

        def gemm():
            history.append('gemm')
            return captured_output

        self_test = self
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            Event=Event, CUDAGraph=Graph, graph=capture,
            synchronize=lambda: history.append('device.synchronize')))
        flush = SimpleNamespace(zero_=lambda: history.append('flush'))
        result = runner.benchmark_graph(
            gemm, SimpleNamespace(warmup=0, num_tests=3), fake_torch, flush)
        self.assertEqual(result[:3], (4., 2., 6.))
        self.assertIs(result[3], captured_output)
        self.assertEqual(event_options, [dict(enable_timing=True, external=True)] * 2)
        capture_start = history.index('capture_enter')
        self.assertEqual(history[capture_start:capture_start + 6], [
            'capture_enter', 'flush', 'start.record', 'gemm', 'end.record', 'capture_exit'])
        self.assertEqual(history.count('gemm'), 4)  # Three eager warmups + one capture.
        self.assertEqual(history.count('replay'), 4)  # One graph warmup + three samples.
        self.assertEqual(history.count('end.synchronize'), 3)


@unittest.skipIf(torch is None, 'PyTorch unavailable; standard-library checks still run')
class QuantizerCpuTests(unittest.TestCase):
    def test_zero_groups_remain_finite_and_reconstruct_zero(self):
        for dtype in (torch.float16, torch.bfloat16):
            codes, scales = runner.quantize_e2m1(torch.zeros(3, 256), torch, dtype)
            self.assertEqual(codes.dtype, torch.uint8)
            self.assertEqual(scales.dtype, dtype)
            self.assertEqual(tuple(scales.shape), (3, 2))
            self.assertTrue(torch.equal(codes, torch.zeros_like(codes)))
            self.assertTrue(torch.equal(scales, torch.ones_like(scales)))

    def test_canonical_codes_and_ties_toward_zero(self):
        magnitudes = [0., .5, 1., 1.5, 2., 3., 4., 6.]
        samples = magnitudes + [-value for value in magnitudes]
        samples += list(runner.MIDPOINTS) + [-value for value in runner.MIDPOINTS]
        weight = torch.tensor(samples + [6.] * (128 - len(samples))).reshape(1, 128)
        expected = list(range(16)) + list(range(7)) + list(range(8, 15))
        for dtype in (torch.float16, torch.bfloat16):
            codes, scales = runner.quantize_e2m1(weight, torch, dtype)
            self.assertEqual(scales.item(), 1.)
            self.assertEqual(codes[0, :len(expected)].tolist(), expected)
            table = torch.tensor(runner.E2M1, dtype=dtype)
            self.assertFalse(torch.signbit(table[0]).item())
            self.assertTrue(torch.signbit(table[8]).item())

    def test_per_group_scales_and_nearest_values_use_stored_scale(self):
        generator = torch.Generator().manual_seed(123)
        weight = torch.randn(3, 256, generator=generator)
        weight[:, 128:] *= 17
        for dtype in (torch.float16, torch.bfloat16):
            codes, scales = runner.quantize_e2m1(weight, torch, dtype)
            self.assertEqual(tuple(codes.shape), (3, 256))
            self.assertTrue(codes.is_contiguous() and scales.is_contiguous())
            self.assertTrue(torch.all(scales[:, 1] > scales[:, 0]).item())
            normalized = weight.reshape(3, 2, 128) / scales.float().unsqueeze(-1)
            magnitudes = torch.tensor(runner.E2M1[:8])
            distances = (normalized.abs().unsqueeze(-1) - magnitudes).abs()
            expected_magnitude = distances.argmin(-1).to(torch.uint8)
            expected_sign = torch.signbit(normalized).to(torch.uint8) * 8
            self.assertTrue(torch.equal(codes.reshape(3, 2, 128), expected_magnitude + expected_sign))

    def test_quantizer_rejects_partial_k_group(self):
        with self.assertRaises(ValueError):
            runner.quantize_e2m1(torch.zeros(2, 129), torch, torch.float16)

    def test_reference_accepts_rounded_a16_output_for_gaussian_inputs(self):
        for dtype in (torch.float16, torch.bfloat16):
            generator = torch.Generator().manual_seed(23)
            a = torch.randn(7, 256, generator=generator).to(dtype)
            weight = torch.randn(11, 256, generator=generator)
            weight[:, 128:] *= 5
            codes_nk, scales = runner.quantize_e2m1(weight, torch, dtype)
            codes_kn = codes_nk.T.contiguous()
            table = torch.tensor(runner.E2M1, dtype=dtype)
            reconstructed_nk = (
                table[codes_nk.long()] * scales.repeat_interleave(128, dim=1)).to(dtype)
            output = (a.float() @ reconstructed_nk.T.float()).to(dtype)
            metrics = runner.check_output(
                a, codes_kn, scales, table, output,
                SimpleNamespace(check_rows=0, check_cols=0), torch)
            self.assertEqual(metrics['checked_rows'], 7)
            self.assertEqual(metrics['checked_cols'], 11)
            self.assertLess(metrics['relative_l2'], .004)
            self.assertLess(metrics['diff'], .00002)

    def test_reference_sampling_still_reduces_over_entire_k(self):
        a = torch.zeros(3, 256, dtype=torch.float16)
        a[:, -1] = 1.
        codes_kn = torch.zeros(256, 5, dtype=torch.uint8)
        codes_kn[-1, :] = 7  # LUT[7] = 6, solely in the final K element.
        scales = torch.tensor([[1., 3.]] * 5, dtype=torch.float16)
        table = torch.tensor(runner.E2M1, dtype=torch.float16)
        output = torch.full((3, 5), 18., dtype=torch.float16)
        args = SimpleNamespace(check_rows=2, check_cols=2)
        metrics = runner.check_output(a, codes_kn, scales, table, output, args, torch)
        self.assertEqual(metrics['relative_l2'], 0.)
        self.assertEqual(metrics['diff'], 0.)
        output[:, -1] = 0.
        metrics = runner.check_output(a, codes_kn, scales, table, output, args, torch)
        self.assertGreater(metrics['relative_l2'], .7)

    def test_reference_scans_unsampled_output_for_nonfinite_values(self):
        a = torch.ones(3, 128, dtype=torch.bfloat16)
        codes_kn = torch.ones(128, 5, dtype=torch.uint8)
        scales = torch.ones(5, 1, dtype=torch.bfloat16)
        table = torch.tensor(runner.E2M1, dtype=torch.bfloat16)
        output = torch.full((3, 5), 64., dtype=torch.bfloat16)
        output[1, 2] = float('nan')
        with self.assertRaisesRegex(AssertionError, 'full output scan'):
            runner.check_output(a, codes_kn, scales, table, output,
                                SimpleNamespace(check_rows=1, check_cols=1), torch)


if __name__ == '__main__':
    unittest.main()
