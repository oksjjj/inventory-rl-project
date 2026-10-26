"""Serial-v0를 한 기간씩 실행하며 환경, 행동, 보상을 출력한다.

원본 코드
    gym.make, env.reset, env.step, CoreEnv 내부 행렬(X, Y, R, S, D, U, P)
    보상 공식은 vendor/gym-invmgmt/gym_invmgmt/core_env.py 의 _STEP과 같다.

이 파일에서 추가한 코드
    원본 값을 읽어서 한글로 풀어 쓰는 출력과, 보상 항목을 다시 더한 검증.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

import gym_invmgmt  # noqa: F401  # 원본 환경 등록

# 과제 요구: goodwill을 켠다. gym.make에 demand_config를 넘기면 원본 기본값을 통째로 바꾼다.
SERIAL_DEMAND_CONFIG = {
    "type": "stationary",
    "base_mu": 20,
    "use_goodwill": True,
}


def node_role(env, node_id: int) -> str:
    network = env.network
    if node_id in network.market:
        return "시장"
    if node_id in network.retail:
        return "소매"
    if node_id in network.factory:
        return "공장"
    if node_id in network.rawmat:
        return "원자재"
    if node_id in network.distrib:
        return "유통"
    return "기타"


def section(title: str) -> None:
    print()
    print(f"[{title}]")
    print("-" * 60)


def print_environment(env) -> None:
    network = env.network
    print("=== 환경 GymInvMgmt/Serial-v0 ===")
    print(f"에피소드 길이: {env.num_periods}기간")
    print(f"관측 차원: {env.observation_space.shape[0]}")
    print(f"행동 차원: {env.action_space.shape[0]}")
    print("원본 관측 STATUS 구성")
    print("  [직전 수요, 직전 미충족, 보유 재고 X, 이동 중 도착 일정, 기간 t, sentiment]")
    engine = env.demand_engine
    print("goodwill")
    print(f"  사용: {engine.use_goodwill}")
    print(f"  시작 sentiment: 1.0, 충족 시 ×{engine.gw_growth}, 미충족 시 ×{engine.gw_decay}")
    print(f"  범위: {engine.gw_floor} ~ {engine.gw_cap}")
    print()
    print("노드")
    for node_id in network.all_nodes:
        attrs = dict(network.graph.nodes[node_id])
        print(f"  {node_id} ({node_role(env, node_id)}): {attrs or '속성 없음'}")
    print()
    print("링크")
    for src, dst, attrs in network.graph.edges(data=True):
        printable = {k: v for k, v in attrs.items() if k != "demand_dist"}
        print(f"  {src} -> {dst}: {printable}")
    print()
    print("행동 한 칸이 의미하는 발주 링크")
    for i, (src, dst) in enumerate(network.reorder_links):
        lead = network.lead_times[(src, dst)]
        print(
            f"  action[{i}] = {src}({node_role(env, src)}) -> "
            f"{dst}({node_role(env, dst)}) 발주량, 리드타임 L={lead}"
        )
    print()
    print("원본 step() 순서")
    print("  1. 발주량을 공급 가능량으로 자른 뒤 R에 기록")
    print("  2. L기간 전 발주가 입고되고 재고 X를 갱신")
    print("  3. 소매 수요 D를 뽑고 판매 S, 미충족 U를 계산")
    print("  4. 노드별 이익 P를 더해 이번 기간 보상으로 반환")
    print()
    print("원본 보상 공식")
    print("  r_t = α^t × (SR − PC − OC − HC − UP − FK)")
    print("  SR = Σ p × S   출하/판매 매출")
    print("  PC = Σ p × R   이번 기간 발주 구매비")
    print("  OC = (o / v) × 공장 출하")
    print("  HC = h × 종료재고 + g × 이동중재고")
    print("  UP = b × 미충족")
    print("  FK = 발주가 있으면 K, 이 환경의 K는 0")
    print()


def inventory_by_node(env, period: int) -> dict[int, float]:
    return {
        node_id: float(env.X[period, env.network.node_map[node_id]])
        for node_id in env.network.main_nodes
    }


def receipts_by_node(env, period: int) -> dict[int, float]:
    receipts = {}
    for node_id in env.network.main_nodes:
        incoming = 0.0
        for _, reorder_idx, lead_time in env.network.pred_reorder_indices[node_id]:
            if period - lead_time >= 0:
                incoming += float(env.R[period - lead_time, reorder_idx])
        receipts[node_id] = incoming
    return receipts


def outflow_by_node(env, period: int) -> dict[int, float]:
    """노드에서 나간 수량. 유통/공장은 하류 출하, 소매는 판매."""
    outflows = {}
    for node_id in env.network.main_nodes:
        shipped = 0.0
        for _, net_idx in env.network.succ_network_indices[node_id]:
            shipped += float(env.S[period, net_idx])
        yield_rate = env.network.graph.nodes[node_id].get("v", 1.0)
        outflows[node_id] = shipped / yield_rate
    return outflows


def pipeline_stock(env, period: int) -> dict[tuple[int, int], float]:
    return {
        edge: float(env.Y[period, env.network.reorder_map[edge]])
        for edge in env.network.reorder_links
    }


def pipeline_arrivals_at_start(env, period: int) -> list[tuple]:
    """기간 시작 관측의 파이프라인. 원본 CoreEnv._update_state와 같다."""
    rows = []
    for edge in env.network.reorder_links:
        lead_time = env.network.lead_times[edge]
        if lead_time == 0:
            continue
        reorder_idx = env.network.reorder_map[edge]
        arrivals = np.zeros(lead_time)
        for delay in range(lead_time):
            order_time = period - lead_time + delay
            if 0 <= order_time < period:
                arrivals[delay] = env.R[order_time, reorder_idx]
        schedule = ", ".join(
            f"{delay}기간 후 {arrivals[delay]:.2f}" for delay in range(lead_time)
        )
        rows.append((edge, arrivals, float(arrivals.sum()), schedule))
    return rows


def last_demand_and_backlog(env, period: int) -> tuple[float, float]:
    if period > 0:
        return float(env.D[period - 1, 0]), float(env.U[period - 1, 0])
    return 0.0, 0.0


def observation_sentiment(env, period: int) -> float:
    """행동 직전에 관측되는 sentiment. 원본은 이전 step 종료 시점 값을 준다."""
    if period == 0:
        return 1.0
    return float(env.GW[period - 1])


def pipeline_rows_after_step(env, period: int) -> list[str]:
    rows = []
    for (src, dst), lead_time in env.network.lead_times.items():
        reorder_idx = env.network.reorder_map[(src, dst)]
        in_transit = float(env.Y[period + 1, reorder_idx])
        arrives_at = period + lead_time
        rows.append(
            f"  {src}->{dst}: 이번 발주 {env.R[period, reorder_idx]:.2f}, "
            f"다음 이동 중 Y={in_transit:.2f}, 이번 발주 도착 예정 기간={arrives_at}"
        )
    return rows


def collect_reward_terms(env, period: int) -> dict:
    """원본 CoreEnv._STEP과 같은 항목을 산식 문자열과 함께 모은다."""
    sr_lines, pc_lines, hc_lines, oc_lines, up_lines, fk_lines = [], [], [], [], [], []
    sales_revenue = purchase_cost = holding_cost = 0.0
    operating_cost = backlog_penalty = fixed_order_cost = 0.0

    for node_id in env.network.main_nodes:
        role = node_role(env, node_id)
        for succ_idx, net_idx in env.network.succ_network_indices[node_id]:
            succ_id = env.network.idx_to_node[succ_idx]
            price = float(env.network.graph.edges[(node_id, succ_id)]["p"])
            shipped = float(env.S[period, net_idx])
            value = price * shipped
            sales_revenue += value
            sr_lines.append(
                f"  {role} {node_id}->{succ_id}: {price:.2f} × {shipped:.2f} = {value:.4f}"
            )

        for pred_idx, reorder_idx, _lead in env.network.pred_reorder_indices[node_id]:
            pred_id = env.network.idx_to_node[pred_idx]
            price = float(env.network.graph.edges[(pred_id, node_id)]["p"])
            ordered = float(env.R[period, reorder_idx])
            value = price * ordered
            purchase_cost += value
            pc_lines.append(
                f"  {role} {pred_id}->{node_id}: {price:.2f} × {ordered:.2f} = {value:.4f}"
            )

        if node_id not in env.network.rawmat:
            node_idx = env.network.node_map[node_id]
            hold_rate = float(env.network.graph.nodes[node_id]["h"])
            on_hand = float(env.X[period + 1, node_idx])
            on_hand_cost = hold_rate * on_hand
            holding_cost += on_hand_cost
            hc_lines.append(
                f"  {role} 보유 {node_id}: {hold_rate:.3f} × {on_hand:.2f} = {on_hand_cost:.4f}"
            )
            for pred_idx, reorder_idx, _lead in env.network.pred_reorder_indices[node_id]:
                pred_id = env.network.idx_to_node[pred_idx]
                pipeline_rate = float(env.network.graph.edges[(pred_id, node_id)]["g"])
                in_transit = float(env.Y[period + 1, reorder_idx])
                pipeline_cost = pipeline_rate * in_transit
                holding_cost += pipeline_cost
                hc_lines.append(
                    f"  {role} 이동 {pred_id}->{node_id}: "
                    f"{pipeline_rate:.3f} × {in_transit:.2f} = {pipeline_cost:.4f}"
                )

        if node_id in env.network.factory:
            shipped = sum(float(env.S[period, net_idx]) for _, net_idx in env.network.succ_network_indices[node_id])
            op_rate = float(env.network.graph.nodes[node_id]["o"])
            yield_rate = float(env.network.graph.nodes[node_id]["v"])
            node_operating = (op_rate / yield_rate) * shipped
            operating_cost += node_operating
            oc_lines.append(
                f"  {role} {node_id}: ({op_rate:.3f} / {yield_rate:.3f}) × {shipped:.2f} = {node_operating:.4f}"
            )

        if node_id in env.network.retail:
            for succ_id in env.network.graph.successors(node_id):
                if (node_id, succ_id) in env.network.retail_map:
                    retail_idx = env.network.retail_map[(node_id, succ_id)]
                    penalty_rate = float(env.network.graph.edges[(node_id, succ_id)]["b"])
                    unmet = float(env.U[period, retail_idx])
                    value = penalty_rate * unmet
                    backlog_penalty += value
                    up_lines.append(
                        f"  {role} {node_id}->{succ_id}: {penalty_rate:.3f} × {unmet:.2f} = {value:.4f}"
                    )

        for pred_idx, reorder_idx, _lead in env.network.pred_reorder_indices[node_id]:
            pred_id = env.network.idx_to_node[pred_idx]
            setup_cost = float(env.network.graph.edges[(pred_id, node_id)].get("K", 0.0))
            ordered = float(env.R[period, reorder_idx])
            value = setup_cost if setup_cost > 0.0 and ordered > 0.0 else 0.0
            fixed_order_cost += value
            fk_lines.append(
                f"  {role} {pred_id}->{node_id}: K={setup_cost:.3f}, R={ordered:.2f} → {value:.4f}"
            )

    discount = float(env.alpha**period)
    reconstructed = discount * (
        sales_revenue - purchase_cost - operating_cost - holding_cost - backlog_penalty - fixed_order_cost
    )
    return {
        "sr": sales_revenue,
        "pc": purchase_cost,
        "oc": operating_cost,
        "hc": holding_cost,
        "up": backlog_penalty,
        "fk": fixed_order_cost,
        "alpha": float(env.alpha),
        "discount": discount,
        "reconstructed_reward": float(reconstructed),
        "sr_lines": sr_lines,
        "pc_lines": pc_lines,
        "hc_lines": hc_lines,
        "oc_lines": oc_lines,
        "up_lines": up_lines,
        "fk_lines": fk_lines,
    }


def print_reward_formula(terms: dict, reward: float, info: dict, period: int) -> None:
    print("공식  r = α^t × (SR − PC − OC − HC − UP − FK)")
    print("매출 SR = Σ p × S")
    for line in terms["sr_lines"]:
        print(line)
    print(f"  SR = {terms['sr']:.4f}")
    print("구매비 PC = Σ p × R")
    for line in terms["pc_lines"]:
        print(line)
    print(f"  PC = {terms['pc']:.4f}")
    print("생산비 OC = (o / v) × 공장 출하")
    for line in terms["oc_lines"] or ["  해당 없음"]:
        print(line)
    print(f"  OC = {terms['oc']:.4f}")
    print("재고유지비 HC = h × X[t+1] + g × Y[t+1]")
    for line in terms["hc_lines"]:
        print(line)
    print(f"  HC = {terms['hc']:.4f}")
    print("미충족 벌금 UP = b × U")
    for line in terms["up_lines"] or ["  해당 없음"]:
        print(line)
    print(f"  UP = {terms['up']:.4f}")
    print("고정발주비 FK")
    for line in terms["fk_lines"]:
        print(line)
    print(f"  FK = {terms['fk']:.4f}")
    print(
        f"r = {terms['alpha']:.2f}^{period} × "
        f"({terms['sr']:.4f} − {terms['pc']:.4f} − {terms['oc']:.4f} − "
        f"{terms['hc']:.4f} − {terms['up']:.4f} − {terms['fk']:.4f})"
    )
    print(
        f"  = {terms['discount']:.4f} × "
        f"{terms['sr'] - terms['pc'] - terms['oc'] - terms['hc'] - terms['up'] - terms['fk']:.4f}"
        f" = {terms['reconstructed_reward']:.4f}"
    )
    print(f"원본 env.step 보상: {reward:.4f}")
    print(f"원본 info.step_reward: {info['step_reward']:.4f}")
    print(f"보상 일치: {np.isclose(terms['reconstructed_reward'], reward)}")


def print_period(env, period: int, action: np.ndarray, reward: float, info: dict) -> None:
    retail_idx = 0
    retail_link = env.network.retail_links[retail_idx]
    retail_net_idx = env.network.network_map[retail_link]
    opening = inventory_by_node(env, period)
    closing = inventory_by_node(env, period + 1)
    receipts = receipts_by_node(env, period)
    outflows = outflow_by_node(env, period)
    terms = collect_reward_terms(env, period)
    demand = float(env.D[period, retail_idx])
    sales = float(env.S[period, retail_net_idx])
    backlog = float(env.U[period, retail_idx])

    print()
    print("=" * 60)
    print(f"기간 {period}")
    print("=" * 60)

    last_demand, last_backlog = last_demand_and_backlog(env, period)
    start_pipeline = pipeline_stock(env, period)

    section("STATUS")
    print("원본 관측이 담는 값")
    print(f"  직전 수요: {last_demand:.2f}")
    print(f"  직전 미충족: {last_backlog:.2f}")
    print(f"  기간 t: {period}")
    print(f"  sentiment: {observation_sentiment(env, period):.3f}")
    print("보유 재고 X[t]")
    for node_id, qty in opening.items():
        print(f"  노드 {node_id} ({node_role(env, node_id)}): {qty:.2f}")
    print("이동 중 재고 Y[t]")
    for (src, dst), qty in start_pipeline.items():
        lead_time = env.network.lead_times[(src, dst)]
        print(f"  {src}->{dst} (L={lead_time}): {qty:.2f}")
    print("도착 일정 (관측 파이프라인, 0기간 후 = 이번 기간 입고)")
    arrival_rows = pipeline_arrivals_at_start(env, period)
    if arrival_rows:
        for (src, dst), _arrivals, total, schedule in arrival_rows:
            print(f"  {src}->{dst}: 합 {total:.2f}  [{schedule}]")
    else:
        print("  없음")

    section("ACTION")
    print("발주 요청량과 실제 반영된 R")
    for i, (src, dst) in enumerate(env.network.reorder_links):
        filled = float(env.R[period, i])
        print(
            f"  action[{i}] {src}->{dst}: "
            f"요청 {action[i]:.2f}  →  실제 R={filled:.2f}"
        )

    section("STATUS 변화")
    print("입고 R[t-L]")
    for node_id, qty in receipts.items():
        print(f"  노드 {node_id} ({node_role(env, node_id)}): {qty:.2f}")
    print(f"소매 수요 D: {demand:.2f}")
    print(f"소매 판매 S: {sales:.2f}")
    print(f"소매 미충족 U: {backlog:.2f}")
    print(
        "goodwill 갱신: "
        f"관측 {observation_sentiment(env, period):.3f} → "
        f"이번 수요에 사용한 값 {float(env.GW[period]):.3f}"
    )
    print("다음 이동 중 재고 Y[t+1]")
    for row in pipeline_rows_after_step(env, period):
        print(row)
    print("다음 상태 재고  X[t+1] = X[t] + 입고 - 출고")
    for node_id in env.network.main_nodes:
        start = opening[node_id]
        incoming = receipts[node_id]
        outgoing = outflows[node_id]
        end = closing[node_id]
        print(
            f"  노드 {node_id} ({node_role(env, node_id)}): "
            f"{start:.2f} + {incoming:.2f} - {outgoing:.2f} = {end:.2f}"
        )

    section("REWARD")
    print_reward_formula(terms, reward, info, period)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serial-v0를 한 기간씩 살펴본다.")
    parser.add_argument("--steps", type=int, default=3, help="출력할 기간 수. 기본 3")
    parser.add_argument("--order", type=float, default=20.0, help="모든 링크에 넣는 고정 발주량. 기본 20")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = gym.make("GymInvMgmt/Serial-v0", demand_config=SERIAL_DEMAND_CONFIG)
    env.reset(seed=args.seed)
    core = env.unwrapped

    print_environment(core)
    action = np.full(core.action_space.shape, args.order, dtype=np.float64)
    print("환경: GymInvMgmt/Serial-v0")
    print(f"이번 실행 정책: 매 기간 모든 링크에 {args.order} 발주")
    print(f"seed={args.seed}, 출력 기간={args.steps}")
    print()

    for _ in range(args.steps):
        period = core.period
        _obs, reward, terminated, truncated, info = env.step(action)
        print_period(core, period, action, float(reward), info)
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
