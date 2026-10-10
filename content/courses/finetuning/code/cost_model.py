"""Cost model for fine-tuning and serving: SFT, GRPO, serving, and build-vs-buy.

Module: 20 — Production: Serving Adapters, MLOps, and Cost Planning (Lab L20.3).
Hardware tier: none (pure arithmetic, stdlib only). There is no GPU "full" mode;
the default run prints a report for your own assumptions, `--smoke` additionally
asserts the module's worked numbers.

Every price and throughput is an ASSUMPTION you pass as a flag. Replace the
defaults with your own quotes and measured throughput before using the output.

Run (from the repository root):
    python content/courses/finetuning/code/cost_model.py --smoke
    python content/courses/finetuning/code/cost_model.py --gpu-price 3.10 --decode-tps 1800
"""
import argparse
import math


def gpu_hours(flops, peak, mfu):
    return flops / (peak * mfu) / 3600.0


def scenario_a(a):
    """LoRA vs full SFT on `sft_tokens` tokens."""
    full = 6 * a.params * a.sft_tokens
    lora = 4 * a.params * a.sft_tokens
    return {
        "full_flops": full,
        "lora_flops": lora,
        "full_gpu_h": gpu_hours(full, a.peak_flops, a.mfu),
        "lora_gpu_h": gpu_hours(lora, a.peak_flops, a.mfu),
    }


def scenario_b(a):
    """GRPO: generation (decode-throughput bound) plus training on rollouts."""
    tokens = a.generations * a.prompts_per_step * a.steps * a.completion_tokens
    gen_h = tokens / a.decode_tps / 3600.0
    per_token = 4 if a.rl_lora else 6
    train_h = gpu_hours(per_token * a.params * tokens, a.peak_flops, a.mfu)
    if a.beta > 0:  # reference-model forward pass
        train_h += gpu_hours(2 * a.params * tokens, a.peak_flops, a.mfu)
    total = gen_h + train_h
    return {"gen_tokens": tokens, "gen_gpu_h": gen_h, "train_gpu_h": train_h,
            "total_gpu_h": total, "gen_share": gen_h / total}


def scenario_c(a):
    """Serving R requests/day at t_in + t_out tokens."""
    req_s = a.requests_per_day / 86400.0
    out_tps = req_s * a.t_out
    gpus_fluid = out_tps * a.peak_factor / a.serve_tps
    gpus = math.ceil(gpus_fluid) + a.redundancy
    day_cost = gpus * 24 * a.gpu_price
    tokens_day = a.requests_per_day * (a.t_in + a.t_out)
    api_req = (a.t_in * a.api_in + a.t_out * a.api_out) / 1e6
    return {"req_per_s": req_s, "out_tps": out_tps, "gpus_fluid": gpus_fluid,
            "gpus": gpus, "self_day": day_cost,
            "self_per_req": day_cost / a.requests_per_day,
            "self_per_1m_tok": day_cost / (tokens_day / 1e6),
            "api_per_req": api_req, "api_day": api_req * a.requests_per_day}


def break_even(a):
    api_req = (a.t_in * a.api_in + a.t_out * a.api_out) / 1e6
    v = a.t_out * a.peak_factor * a.gpu_price / (a.serve_tps * 3600.0)
    fixed = a.ops_monthly + a.train_cost / a.amortize_months
    r_star = fixed / (30 * (api_req - v)) if api_req > v else float("inf")

    def months(r):
        return 30 * r * api_req, fixed + 30 * r * v

    return {"a": api_req, "v": v, "F": fixed, "R_star": r_star, "months": months}


def reconcile(a):
    """Monthly GPU spend at `requests_per_day`: fluid break-even model vs scenario (c).

    The fluid model bills fractional GPUs; scenario (c) rounds up and adds N+1 spares.
    """
    v = break_even(a)["v"]
    rc = scenario_c(a)
    fluid = 30 * a.requests_per_day * v
    integer = 30 * math.ceil(rc["gpus_fluid"]) * 24 * a.gpu_price
    spare = 30 * a.redundancy * 24 * a.gpu_price
    return {"fluid_month": fluid, "rounding_month": integer - fluid,
            "redundancy_month": spare, "scenario_c_month": 30 * rc["self_day"]}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true", help="assert the module's worked numbers")
    p.add_argument("--params", type=float, default=8e9)
    p.add_argument("--peak-flops", type=float, default=989e12, help="dense bf16 FLOP/s (H100 SXM)")
    p.add_argument("--mfu", type=float, default=0.40)
    p.add_argument("--gpu-price", type=float, default=2.50, help="USD per GPU-hour (assumption)")
    p.add_argument("--sft-tokens", type=float, default=5e7)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=512)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--completion-tokens", type=int, default=1024)
    p.add_argument("--decode-tps", type=float, default=2500, help="decode tokens/s/GPU (assumption)")
    p.add_argument("--rl-lora", action="store_true", help="use 4ND instead of 6ND for RL training")
    p.add_argument("--beta", type=float, default=0.0, help="KL coefficient; >0 adds a reference forward")
    p.add_argument("--requests-per-day", type=float, default=1e6)
    p.add_argument("--t-in", type=int, default=500)
    p.add_argument("--t-out", type=int, default=200)
    p.add_argument("--serve-tps", type=float, default=3000, help="output tokens/s/GPU at SLO (assumption)")
    p.add_argument("--peak-factor", type=float, default=2.0)
    p.add_argument("--redundancy", type=int, default=1)
    p.add_argument("--api-in", type=float, default=0.40, help="USD per 1M input tokens (assumption)")
    p.add_argument("--api-out", type=float, default=1.60, help="USD per 1M output tokens (assumption)")
    p.add_argument("--ops-monthly", type=float, default=5000.0)
    p.add_argument("--train-cost", type=float, default=3000.0)
    p.add_argument("--amortize-months", type=float, default=6.0)
    return p


def report(a):
    ra, rb, rc, be = scenario_a(a), scenario_b(a), scenario_c(a), break_even(a)
    print("(a) SFT: full %.3g FLOPs = %.3f GPU-h (USD %.2f); LoRA %.3g FLOPs = %.3f GPU-h (USD %.2f)" % (
        ra["full_flops"], ra["full_gpu_h"], ra["full_gpu_h"] * a.gpu_price,
        ra["lora_flops"], ra["lora_gpu_h"], ra["lora_gpu_h"] * a.gpu_price))
    print("(b) GRPO: %d generated tokens; generation %.1f GPU-h + training %.1f GPU-h = %.1f "
          "(USD %.0f); generation share %.0f%%" % (
              rb["gen_tokens"], rb["gen_gpu_h"], rb["train_gpu_h"], rb["total_gpu_h"],
              rb["total_gpu_h"] * a.gpu_price, 100 * rb["gen_share"]))
    print("(c) Serving: %.2f req/s, %.0f out tok/s, %.2f GPUs fluid -> %d GPUs; USD %.0f/day "
          "(USD %.5f/req, USD %.3f per 1M tok) vs API USD %.0f/day (USD %.5f/req)" % (
              rc["req_per_s"], rc["out_tps"], rc["gpus_fluid"], rc["gpus"], rc["self_day"],
              rc["self_per_req"], rc["self_per_1m_tok"], rc["api_day"], rc["api_per_req"]))
    print("Break-even: a=USD %.6f/req, v=USD %.7f/req, F=USD %.0f/month -> R*=%.0f req/day" % (
        be["a"], be["v"], be["F"], be["R_star"]))
    for r in (1e5, 1e6, 1e7):
        api_m, self_m = be["months"](r)
        print("  %10.0f req/day: API USD %10.2f/month, self-host USD %10.2f/month" % (r, api_m, self_m))
    rec = reconcile(a)
    print("GPU spend/month at %.0f req/day: fluid USD %.2f + integer rounding USD %.2f + N+1 spare "
          "USD %.2f = scenario (c) USD %.2f" % (a.requests_per_day, rec["fluid_month"],
                                               rec["rounding_month"], rec["redundancy_month"],
                                               rec["scenario_c_month"]))
    return ra, rb, rc, be


def smoke_asserts(a):
    defaults = build_parser().parse_args([])
    ra, rb, rc, be = (scenario_a(defaults), scenario_b(defaults),
                      scenario_c(defaults), break_even(defaults))
    assert abs(ra["full_flops"] - 2.4e18) < 1e6 and abs(ra["lora_flops"] - 1.6e18) < 1e6
    assert round(ra["full_gpu_h"], 3) == 1.685 and round(ra["lora_gpu_h"], 3) == 1.123
    assert rb["gen_tokens"] == 2_097_152_000
    assert round(rb["gen_gpu_h"], 1) == 233.0 and round(rb["train_gpu_h"], 1) == 70.7
    assert rb["gen_gpu_h"] > rb["train_gpu_h"] and round(rb["gen_share"], 2) == 0.77
    assert rc["gpus"] == 3 and round(rc["self_day"]) == 180 and round(rc["api_day"]) == 520
    assert round(be["R_star"]) == 428943
    assert [tuple(round(x, 2) for x in be["months"](r)) for r in (1e5, 1e6, 1e7)] == [
        (1560.0, 5777.78), (15600.0, 8277.78), (156000.0, 33277.78)]
    rec = reconcile(defaults)
    assert round(rec["fluid_month"], 2) == 2777.78 and round(rec["scenario_c_month"]) == 5400
    assert round(rec["rounding_month"], 2) == 822.22 and round(rec["redundancy_month"]) == 1800
    assert abs(rec["fluid_month"] + rec["rounding_month"] + rec["redundancy_month"]
               - rec["scenario_c_month"]) < 1e-6
    print("smoke: all worked numbers from module 20 verified")


def main():
    args = build_parser().parse_args()
    report(args)
    if args.smoke:
        smoke_asserts(args)


if __name__ == "__main__":
    main()
