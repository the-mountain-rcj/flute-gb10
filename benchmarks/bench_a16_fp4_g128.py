"""Stock FLUTE A16 x E2M1 FP4, one 16-bit scale per 128 K elements.

CSV shape convention: A[M,K], B[N,K], D=A@B.T. No kernel modifications.
Only --dry-run and the CSV helpers require no PyTorch/CUDA installation.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


GROUP_SIZE = 128
E2M1 = (0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.)
MIDPOINTS = (.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)
RESULT_FIELDS = [
    'testcase_id', 'm', 'k', 'n', 'passed', 'diff', 'max_abs',
    'elapsed_us', 'tflops', 'gbps', 'error',
    'backend', 'a_dtype', 'weight_format', 'group_size', 'scale_dtype',
    'template_id', 'num_sms', 'checked_rows', 'checked_cols', 'relative_l2',
    'min_us', 'max_us', 'baseline_us', 'speedup_vs_a16',
]


def read_cases(path):
    if Path(path).suffix.lower() in ('.xlsx', '.xls', '.xlsm'):
        raise ValueError('Export Excel as CSV UTF-8: testcase_id,m,k,n')
    cases = []
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        names = [name.strip() for name in (reader.fieldnames or [])]
        missing = {'testcase_id', 'm', 'k', 'n'} - set(names)
        if missing:
            raise ValueError(f'CSV missing columns: {sorted(missing)}')
        if len(names) != len(set(names)):
            raise ValueError('CSV contains duplicate column names')
        for row in reader:
            if None in row:
                raise ValueError(f'CSV row {reader.line_num}: more values than columns')
            row = {key.strip(): (value or '').strip() for key, value in row.items()}
            if not any(row.values()):
                continue
            try:
                if not row['testcase_id']:
                    raise ValueError('empty testcase_id')
                cases.append(dict(testcase_id=row['testcase_id'],
                                  **{key: int(row[key]) for key in ('m', 'k', 'n')}))
            except ValueError as exc:
                raise ValueError(f'CSV row {reader.line_num}: {exc}') from exc
    if not cases:
        raise ValueError('CSV contains no cases')
    return cases


def validate_case(case):
    if any(case[key] <= 0 for key in ('m', 'k', 'n')):
        raise ValueError('M, K, N must be positive integers')
    if case['k'] % 512:
        raise ValueError('K must be a multiple of 512 (128-element groups, safe vectorized scale loads)')
    if case['n'] % 256:
        raise ValueError('N must be a multiple of 256 (safe for all current FLUTE W4 packing templates)')


def selected_indices(size, limit):
    """Even coverage, including both ends; 0 means every element."""
    if limit == 0 or limit >= size:
        return list(range(size))
    if limit == 1:
        return [0]
    return [i * (size - 1) // (limit - 1) for i in range(limit)]


def logical_bytes(m, k, n):
    # A16 + packed W4 + per-group A16 scales + output A16. Excludes scratch/LUT.
    return 2 * m * k + n * k // 2 + 2 * n * (k // GROUP_SIZE) + 2 * m * n


def quantize_e2m1(weight, torch, dtype):
    """Offline absmax/6 quantization. Input B[N,K]; return codes[N,K], S[N,K/128].

    Scales are rounded to their actual storage dtype BEFORE code selection.
    Nearest E2M1 magnitude, midpoint ties toward zero; no zero point/global scale.
    This is a benchmark quantizer, not a model calibration method.
    """
    if weight.ndim != 2 or weight.shape[1] % GROUP_SIZE:
        raise ValueError('Expected weight[N,K] with K divisible by 128')
    grouped = weight.float().reshape(weight.shape[0], -1, GROUP_SIZE)
    maxima = grouped.abs().amax(dim=-1)
    scales = (maxima / 6).clamp_min(torch.finfo(dtype).tiny).to(dtype)
    scales = torch.where(maxima == 0, torch.ones_like(scales), scales)
    normalized = grouped / scales.float().unsqueeze(-1)
    boundaries = torch.tensor(MIDPOINTS, device=weight.device, dtype=torch.float32)
    codes = torch.bucketize(normalized.abs().contiguous(), boundaries).to(torch.uint8)
    codes = codes + torch.signbit(normalized).to(torch.uint8) * 8
    return codes.reshape(weight.shape).contiguous(), scales.contiguous()


def load_runtime(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA PyTorch required; use --dry-run for CSV validation only')
    torch.cuda.set_device(args.device)
    cap = torch.cuda.get_device_capability(args.device)
    actual_sm = f'{cap[0]}{cap[1]}'
    if actual_sm != args.target_sm:
        raise RuntimeError(f'Expected SM{args.target_sm}, got SM{actual_sm}; select --target-sm explicitly')
    import flute
    import flute.tune
    import flute.utils
    if not callable(getattr(flute.tune, 'tune_and_pack', None)) or not flute.TEMPLATE_CONFIGS:
        raise RuntimeError('Require current stock FLUTE tune_and_pack and template config data')
    dtype = torch.bfloat16 if args.a_dtype == 'bf16' else torch.float16
    return SimpleNamespace(torch=torch, flute=flute, dtype=dtype,
                           device=torch.device('cuda', args.device))


def check_output(a, codes_kn, scales, table, output, args, torch):
    """Sample rows/columns across the output, always sum over the entire K."""
    if not torch.isfinite(output).all().item():
        raise AssertionError('Output contains NaN/Inf (full output scan)')
    rows = selected_indices(a.shape[0], args.check_rows)
    cols = selected_indices(scales.shape[0], args.check_cols)
    sums = [0., 0., 0., 0.]
    max_abs = 0.
    for col_start in range(0, len(cols), 128):
        col_ids = torch.tensor(cols[col_start:col_start + 128], device=a.device)
        codes = codes_kn.index_select(1, col_ids).long()
        scale = scales.index_select(0, col_ids).repeat_interleave(GROUP_SIZE, dim=1).T
        # Match the 16-bit dequantization performed inside the W4A16 kernel.
        weight = (table[codes] * scale).to(a.dtype).float()
        for row_start in range(0, len(rows), 256):
            row_ids = torch.tensor(rows[row_start:row_start + 256], device=a.device)
            ref = a.index_select(0, row_ids).float() @ weight
            got = output.index_select(0, row_ids).index_select(1, col_ids).float()
            delta = got - ref
            sums[0] += (got * got).sum(dtype=torch.float64).item()
            sums[1] += (ref * ref).sum(dtype=torch.float64).item()
            sums[2] += (got * ref).sum(dtype=torch.float64).item()
            sums[3] += (delta * delta).sum(dtype=torch.float64).item()
            max_abs = max(max_abs, delta.abs().max().item())
    denominator = sums[0] + sums[1]
    diff = max(0., 1 - 2 * sums[2] / denominator) if denominator else 0.
    relative_l2 = math.sqrt(sums[3] / sums[1]) if sums[1] else (0. if sums[3] == 0 else math.inf)
    return dict(diff=diff, max_abs=max_abs, relative_l2=relative_l2,
                checked_rows=len(rows), checked_cols=len(cols))


def benchmark_graph(fn, args, torch, flush):
    """One stock call in a graph, external CUDA event nodes around ONLY the call.

    Cache flush is captured before the start event. No Python enqueue gap between
    timer events. Includes all GEMM/dequantization/reduction GPU work in the call.
    """
    for _ in range(max(3, args.warmup)):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if flush is not None:
            flush.zero_()
        start.record()
        output = fn()
        end.record()
    for _ in range(max(1, args.warmup)):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(args.num_tests):
        graph.replay()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    if any(not math.isfinite(t) or t <= 0 for t in times):
        raise RuntimeError(f'Invalid CUDA event timings: {times}')
    return statistics.median(times), min(times), max(times), output


def run_case(case, args, runtime):
    result = dict.fromkeys(RESULT_FIELDS, '')
    result.update(case, passed=False, backend='flute', a_dtype=args.a_dtype,
                  scale_dtype=args.a_dtype, weight_format='fp4_e2m1', group_size=GROUP_SIZE)
    torch, flute = runtime.torch, runtime.flute
    try:
        validate_case(case)
        m, k, n = (case[key] for key in ('m', 'k', 'n'))
        torch.manual_seed(args.seed)
        a = torch.randn((m, k), device=runtime.device, dtype=runtime.dtype)
        codes_kn = torch.empty((k, n), device=runtime.device, dtype=torch.uint8)
        scales = torch.empty((n, k // GROUP_SIZE), device=runtime.device, dtype=runtime.dtype)
        for offset in range(0, n, 256):
            weight = torch.randn((min(256, n - offset), k), device=runtime.device)
            codes, scale = quantize_e2m1(weight, torch, runtime.dtype)
            codes_kn[:, offset:offset + codes.shape[0]] = codes.T
            scales[offset:offset + codes.shape[0]] = scale
        del weight, codes, scale
        table = torch.tensor(E2M1, device=runtime.device, dtype=runtime.dtype)
        table2 = flute.utils.make_qmap2_from_qmap(table)
        print(f" > tuning id={case['testcase_id']} (offline; may take minutes)", flush=True)
        packed, meta = flute.tune.tune_and_pack(
            a, codes_kn, num_bits=4, group_size=GROUP_SIZE,
            num_seeds=args.tune_seeds, check_correctness=False)
        config = flute.utils.get_template_config(4, meta.template_id, meta.num_sms)
        if n % (4 * config['tileP']):
            raise RuntimeError('Selected template has incompatible N packing alignment')
        if packed.dtype != torch.int16 or packed.shape != (n // 4, k) or not packed.is_contiguous():
            raise RuntimeError('Unexpected FLUTE packed weight layout')
        result.update(template_id=meta.template_id, num_sms=meta.num_sms)
        workspace = flute.utils.make_workspace_streamk(runtime.device)

        def gemm():
            return flute.tune.qgemm_v2(a, packed, scales, table, table2, workspace, meta)

        output = gemm()
        metrics = check_output(a, codes_kn, scales, table, output, args, torch)
        result.update(metrics)
        if (not math.isfinite(metrics['diff']) or metrics['diff'] >= args.max_diff or
                not math.isfinite(metrics['relative_l2']) or metrics['relative_l2'] >= args.max_relative_l2):
            raise AssertionError(f"Kernel correctness diff={metrics['diff']} (limit {args.max_diff}), "
                                 f"relative_l2={metrics['relative_l2']} (limit {args.max_relative_l2})")
        del output
        flush = None if args.no_flush_l2 else torch.empty(
            args.flush_mb * 1024 * 1024, device=runtime.device, dtype=torch.uint8)
        elapsed, minimum, maximum, output = benchmark_graph(gemm, args, torch, flush)
        # Stream-K uses shared scratch and can be nondeterministic: also check
        # the final graph replay, not only a one-shot eager invocation.
        metrics2 = check_output(a, codes_kn, scales, table, output, args, torch)
        result.update({key: max(metrics[key], metrics2[key]) for key in metrics})
        if (not math.isfinite(metrics2['diff']) or metrics2['diff'] >= args.max_diff or
                not math.isfinite(metrics2['relative_l2']) or metrics2['relative_l2'] >= args.max_relative_l2):
            raise AssertionError(f"After graph replay: diff={metrics2['diff']} (limit {args.max_diff}), "
                                 f"relative_l2={metrics2['relative_l2']} (limit {args.max_relative_l2})")
        if args.baseline:
            # This intentionally dequantizes OUTSIDE timing: labeled A16 baseline,
            # never reported as FP4 kernel performance.
            dense = torch.empty((n, k), device=runtime.device, dtype=runtime.dtype)
            for offset in range(0, n, 256):
                ids = codes_kn[:, offset:offset + 256].T.long()
                dense[offset:offset + 256] = table[ids] * scales[offset:offset + 256].repeat_interleave(GROUP_SIZE, dim=1)
            dense_out = torch.empty((m, n), device=runtime.device, dtype=runtime.dtype)
            baseline, _, _, _ = benchmark_graph(lambda: torch.mm(a, dense.T, out=dense_out), args, torch, flush)
            result.update(baseline_us=baseline, speedup_vs_a16=baseline / elapsed)
        result.update(passed=True, elapsed_us=elapsed, min_us=minimum, max_us=maximum,
                      tflops=2 * m * n * k / (elapsed * 1e6),
                      gbps=logical_bytes(m, k, n) / (elapsed * 1000))
    except Exception as exc:
        result['error'] = repr(exc)
    return result


def git_revision(path):
    try:
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'


def write_results(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def write_metadata(args, runtime):
    props = runtime.torch.cuda.get_device_properties(runtime.device)
    data = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), arguments=vars(args),
        runner_revision=git_revision(Path(__file__).resolve().parents[1]),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        flute_path=str(runtime.flute.__file__),
        flute_revision=git_revision(Path(runtime.flute.__file__).resolve().parent),
        torch_version=str(runtime.torch.__version__), cuda=runtime.torch.version.cuda,
        python=platform.python_version(), platform=platform.platform(), gpu=props.name,
        compute_capability=[props.major, props.minor], num_sms=props.multi_processor_count,
        operation='D=A[M,K] @ B[N,K].T, dense FLOPs=2*M*N*K',
        weight_format='E2M1 LUT, actual packed 4-bit weights; NOT INT4/NF4/NVFP4/MXFP4',
        group_size=GROUP_SIZE, group_axis='K', scale_dtype=args.a_dtype, lut=E2M1,
        quantization='absmax/6, stored 16-bit scales, nearest E2M1; ties toward zero; offline synthetic weights',
        timing='median external CUDA event node interval in a one-call CUDA graph; includes fused dequant/MMA/Stream-K; excludes tuning/packing/cache flush/host allocation',
        cache='hot' if args.no_flush_l2 else f'{args.flush_mb} MiB write before every measured call',
        correctness='sampled output rows/columns across full K vs FP32 matmul of dequantized 16-bit weights; full output NaN/Inf scan; checked before and after replay; not quantization error vs original weights',
        diff='1 - 2*dot(got,ref)/(norm(got)^2+norm(ref)^2); same metric family as previous runner; reference differs',
        gbps='logical A + W4 + 16-bit scales + D bytes / time; NOT measured DRAM bandwidth; excludes scratch/LUT',
        baseline='optional torch.mm A16 weights, dequantized outside timing; not a W4A16 performance result',
    )
    path = Path(args.result_csv).with_suffix('.metadata.json')
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--result-csv', default='a16_fp4_g128_result.csv')
    parser.add_argument('--a-dtype', choices=('bf16', 'fp16'), default='bf16')
    parser.add_argument('--target-sm', default='121', help='Expected CUDA capability, e.g. 121 or 100')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--num-tests', type=int, default=30)
    parser.add_argument('--tune-seeds', type=int, default=1)
    parser.add_argument('--max-diff', type=float, default=None)
    parser.add_argument('--max-relative-l2', type=float, default=None)
    parser.add_argument('--check-rows', type=int, default=32, help='0 checks every row')
    parser.add_argument('--check-cols', type=int, default=512, help='0 checks every column')
    parser.add_argument('--flush-mb', type=int, default=256, help='MiB cache-flush allocation; excluded from timing')
    parser.add_argument('--no-flush-l2', action='store_true')
    parser.add_argument('--baseline', action='store_true', help='Also time torch.mm with already dequantized A16 weights')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.max_diff is None:
        args.max_diff = 0.0001 if args.a_dtype == 'bf16' else 0.000003
    if args.max_relative_l2 is None:
        args.max_relative_l2 = 0.011 if args.a_dtype == 'bf16' else 0.002
    if (min(args.num_tests, args.tune_seeds, args.flush_mb) <= 0 or
            min(args.warmup, args.check_rows, args.check_cols, args.device) < 0 or
            not math.isfinite(args.max_diff) or args.max_diff <= 0 or
            not math.isfinite(args.max_relative_l2) or args.max_relative_l2 <= 0 or
            not args.target_sm.isdigit()):
        parser.error('Invalid positive counts, nonnegative indices, target SM or finite max-diff')
    if Path(args.csv).resolve() in (Path(args.result_csv).resolve(), Path(args.result_csv).with_suffix('.metadata.json').resolve()):
        parser.error('Input CSV must not be overwritten by results or metadata')
    try:
        cases = read_cases(args.csv)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        failures = 0
        for case in cases:
            try:
                validate_case(case)
                status = 'VALID'
            except ValueError as exc:
                status = f'INVALID: {exc}'
                failures += 1
            print(f" > {case['testcase_id']}: M={case['m']} K={case['k']} N={case['n']}: {status}")
        print(f' > {len(cases) - failures}/{len(cases)} shapes valid; NO GPU verification performed')
        return int(failures > 0)
    try:
        runtime = load_runtime(args)
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, f'Environment error: {exc}\nSee docs/gb10_a16_fp4_g128.md for source build steps.\n')
    torch = runtime.torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    write_results(args.result_csv, [])
    write_metadata(args, runtime)
    print(f' > GPU: {torch.cuda.get_device_name()}, SM{args.target_sm}; FLUTE: {runtime.flute.__file__}')
    print(f' > A={args.a_dtype}; W=E2M1; K group=128; scale={args.a_dtype}; median CUDA-graph timing')
    results = []
    with torch.inference_mode():
        for index, case in enumerate(cases, 1):
            result = run_case(case, args, runtime)
            results.append(result)
            write_results(args.result_csv, results)
            print(f" > [{index}/{len(cases)}] {case['testcase_id']}: passed={result['passed']}, "
                  f"us={result['elapsed_us']}, TFLOPS={result['tflops']}, diff={result['diff']}, error={result['error']}", flush=True)
            if 'CUDA error' in result['error']:
                print(' > Stopping after CUDA error; restart this process before retrying.')
                break
    passed = sum(bool(row['passed']) for row in results)
    print(f' > {passed}/{len(cases)} passed; results: {args.result_csv}')
    return int(passed != len(cases))


if __name__ == '__main__':
    sys.exit(main())
