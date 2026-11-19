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

# gym.make에 넘기면 원본 Serial-v0 기본 수요(goodwill 꺼짐)를 통째로 바꾼다.
# type: 시간에 따라 평균이 안 바뀜. 학습 기본값 trend+shock 와 다름.
# base_mu: Poisson 수요의 기본 평균.
# use_goodwill: 품절이면 다음 수요 평균에 sentiment 를 곱함 (과제 요구).
SERIAL_DEMAND_CONFIG = {
    "type": "stationary",
    "base_mu": 20,
    "use_goodwill": True,
}


def node_role(env, node_id: int) -> str:
    """노드 번호 → 한글 역할. 출력용이며 학습에는 안 쓴다.

    network.market 등은 노드 번호 list 이다. in 은 그 목록에 있는지만 본다.
    Serial-v0: 0 시장, 1 소매, 2 유통, 3 공장, 4 원자재.
    """
    network = env.network
    if node_id in network.market:  # [0]
        return "시장"
    if node_id in network.retail:  # [1]
        return "소매"
    if node_id in network.factory:  # [3]
        return "공장"
    if node_id in network.rawmat:  # [4]
        return "원자재"
    if node_id in network.distrib:  # [2]
        return "유통"
    return "기타"


def section(title: str) -> None:
    print()
    print(f"[{title}]")
    print("-" * 60)


def print_environment(env) -> None:
    """reset/step 결과가 아니라, 환경이 들고 있는 공급망·수요 설정을 출력한다."""
    # SupplyChainNetwork. serial.yaml 의 노드·링크. 재고 수량 X 가 아님.
    network = env.network
    print("=== 환경 GymInvMgmt/Serial-v0 ===")
    print(f"에피소드 길이: {env.num_periods}기간")
    print(f"관측 차원: {env.observation_space.shape[0]}")
    print(f"행동 차원: {env.action_space.shape[0]}")
    print("원본 관측 STATUS 구성")
    print("  [직전 수요, 직전 미충족, 보유 재고 X, 이동 중 도착 일정, 기간 t, sentiment]")
    # DemandEngine. 매 기간 수요를 뽑고 goodwill(sentiment)을 갱신한다.
    engine = env.demand_engine
    print("goodwill")
    print(f"  사용: {engine.use_goodwill}")
    print(f"  시작 sentiment: 1.0, 충족 시 ×{engine.gw_growth}, 미충족 시 ×{engine.gw_decay}")
    print(f"  범위: {engine.gw_floor} ~ {engine.gw_cap}")
    print()
    print("노드")
    # I0 시작재고, h 창고유지비, C 공장용량, o 운영비, v 수율. 시장·원자재는 속성 없음.
    for node_id in network.all_nodes:
        attrs = dict(network.graph.nodes[node_id])
        print(f"  {node_id} ({node_role(env, node_id)}): {attrs or '속성 없음'}")
    print()
    print("링크")
    # p 가격, b 품절벌금, L 리드타임, g 파이프라인 유지비(받는 쪽, 매 기간 Y에 부과).
    # demand_dist 는 scipy Poisson 객체. 주소 0x... 는 의미 없고 분포 종류만 보면 된다.
    for src, dst, attrs in network.graph.edges(data=True):
        printable = {k: v for k, v in attrs.items()}
        print(f"  {src} -> {dst}: {printable}")
    print()
    print("행동 한 칸이 의미하는 발주 링크")
    # action[i] = reorder_links[i] 발주량. 1→0 판매 링크는 행동에 없음.
    for i, (src, dst) in enumerate(network.reorder_links):
        lead = network.lead_times[(src, dst)]
        print(
            f"  action[{i}] = {src}({node_role(env, src)}) -> "
            f"{dst}({node_role(env, dst)}) 발주량, 리드타임 L={lead}"
        )
    print()
    print("원본 step() 순서")
    print("  한 기간에 에이전트가 발주 3칸을 넣으면, 아래 순서로 세상이 진행된다.")
    print("  1. 발주량을 공급 가능량으로 자른 뒤 R에 기록")
    print("     R (Reorder): 이번 기간에 실제로 나간 발주량")
    print("     요청 전부가 아니라, 윗단계 재고·공장 용량으로 자른 양")
    print("  2. L기간 전 발주가 입고되고 재고 X를 갱신")
    print("     L (Lead time): 발주가 도착하는 데 걸리는 기간")
    print("     X: 창고에 있는 재고. 입고(+) 후 아래로 보낸 양(−)")
    print("     오늘 발주는 L=4이면 4기간 뒤에 도착. L=0만 같은 기간에 입고")
    print("  3. 소매 수요 D를 뽑고 판매 S, 미충족 U를 계산")
    print("     D (Demand): 고객이 원한 양")
    print("     S (Sales/Shipment): 재고가 있어서 실제로 판(보낸) 양")
    print("     U (Unfulfilled): 못 준 양 = 수요 − 판매")
    print("  4. 노드별 이익 P를 더해 이번 기간 보상으로 반환")
    print("     P (Profit): 소매·유통·공장 각각의 오늘 이익")
    print("     한 기간 보상 r = 세 노드 P의 합. 시장·원자재는 이익을 안 계산")
    print()
    print("원본 보상 공식")
    print("  한 노드 이익 P = α^t × (SR − PC − OC − HC − UP − FK)")
    print("  한 기간 보상 r_t = P_소매 + P_유통 + P_공장")
    print("  오늘 번 돈 = 판 돈 − 오늘 나간 비용들")
    print("  r_t : 기간 t의 보상 (reward)")
    print("  α   : 할인율 (alpha). Serial-v0 기본값 1 → 할인 없음, α^t = 1")
    print("        그래서 여기선 r_t = (소매 이익) + (유통 이익) + (공장 이익)")
    print()
    print("  SR (Sales Revenue, 매출)     = Σ 가격p × 판매S")
    print("  PC (Purchase Cost, 구매비)   = Σ 가격p × 발주R")
    print("  OC (Operating Cost, 운영비)  = (o / v) × 공장 출하   공장만")
    print("      o: 원자재 1단위 가공 비용, v: 원자재→완제품 수율. 여기 v=1이라 OC=0.01×출하")
    print("  HC (Holding Cost, 유지비)    = h×창고재고 + g×파이프라인(오는 중)")
    print("      h: 창고 유지비, g: 이동 중 유지비. 파이프라인 비용은 받는 쪽 노드에 귀속")
    print("  UP (Unfulfilled Penalty, 품절벌금) = b × 미충족U   소매만")
    print("  FK (Fixed order cost K, 고정발주비) = 발주가 있으면 K")
    print("      이 Serial 토폴로지는 K=0 이라 FK는 항상 0")
    print()


def inventory_by_node(env, period: int) -> dict[int, float]:
    """기간 period 의 창고 재고 X. main_nodes = 소매1·유통2·공장3 (시장·원자재 제외).

    X 열 번호와 노드 id 가 다를 수 있어 node_map 으로 칸을 찾는다.
    """
    return {
        # node_map[1]=1 이면 X[period, 1] 이 소매 창고. 예: period=0 → 100
        node_id: float(env.X[period, env.network.node_map[node_id]])
        for node_id in env.network.main_nodes  # 1, 2, 3
    }


def receipts_by_node(env, period: int) -> dict[int, float]:
    """오늘(period) 각 창고로 들어온 입고량. 오늘 발주가 아니라 L기간 전 발주가 도착한 양.

    period: step 직전의 core.period. 0~29. '발주 다음날'이 아님.
    reorder_idx: 발주 링크 번호 = action 번호 (0 유통→소매, 1 공장→유통, 2 원자재→공장).
    R[period - L, reorder_idx]: 그 링크에서 L기간 전에 시킨 양. 그게 오늘 입고.
    period - L < 0 이면 에피소드 시작 전이라 입고 0.
    """
    receipts = {}
    for node_id in env.network.main_nodes:
        incoming = 0.0
        # 이 노드로 들어오는 발주 링크. 소매면 (reorder_idx=0, L=4)
        for _, reorder_idx, lead_time in env.network.pred_reorder_indices[node_id]:
            if period - lead_time >= 0:
                # 예: period=5, L=4 → R[1, 열] 이 오늘 입고
                incoming += float(env.R[period - lead_time, reorder_idx])
        receipts[node_id] = incoming
    return receipts


def outflow_by_node(env, period: int) -> dict[int, float]:
    """노드에서 나간 수량. 유통/공장은 하류 출하, 소매는 판매."""
    outflows = {}
    # 소매1, 유통2, 공장3
    for node_id in env.network.main_nodes:
        shipped = 0.0
        # 이 노드에서 아래로 나가는 링크. net_idx 는 S 의 열 번호 (0~3)
        for _, net_idx in env.network.succ_network_indices[node_id]:
            shipped += float(env.S[period, net_idx])  # 오늘 그 링크로 보낸 양
        yield_rate = env.network.graph.nodes[node_id].get("v", 1.0)
        outflows[node_id] = shipped / yield_rate  # Serial v=1 이라 shipped 그대로
    return outflows


def pipeline_stock(env, period: int) -> dict[tuple[int, int], float]:
    """링크별 이동 중 재고 Y. 유지비 g 는 이 양에 매 기간, 받는 쪽 노드 이익에서 깎인다."""
    return {
        # edge=(2,1) 이면 reorder_map=0. Y[period, 0] 이 유통→소매 이동 중 합
        edge: float(env.Y[period, env.network.reorder_map[edge]])
        for edge in env.network.reorder_links  # (2,1), (3,2), (4,3)
    }


def pipeline_arrivals_at_start(env, period: int) -> list[tuple]:
    """기간 시작 관측의 파이프라인. 원본 CoreEnv._update_state와 같다."""
    rows = []
    # 발주 링크 3개: (2,1), (3,2), (4,3)
    for edge in env.network.reorder_links:
        lead_time = env.network.lead_times[edge]  # (2,1)=4, (3,2)=4, (4,3)=0
        if lead_time == 0:
            continue  # (4,3) 건너뜀. 같은 기간에 도착해서 일정이 없음
        reorder_idx = env.network.reorder_map[edge]  # (2,1)=0, (3,2)=1. R 의 열
        arrivals = np.zeros(lead_time)  # L=4 면 [0, 0, 0, 0]
        for delay in range(lead_time):  # L=4 면 delay=0, 1, 2, 3
            # period=10, L=4: delay=0 → 6, delay=1 → 7, delay=2 → 8, delay=3 → 9
            order_time = period - lead_time + delay
            # 에피소드 시작 전(<0)이거나 오늘 발주(==period)면 0 으로 남김
            if 0 <= order_time < period:
                arrivals[delay] = env.R[order_time, reorder_idx]
        # 출력용: "0기간 후 12.00, 1기간 후 8.00"
        schedule = ", ".join(
            f"{delay}기간 후 {arrivals[delay]:.2f}" for delay in range(lead_time)
        )
        rows.append((edge, arrivals, float(arrivals.sum()), schedule))
    return rows  # Serial 에서는 링크 2개. (4,3) 없음


def last_demand_and_backlog(env, period: int) -> tuple[float, float]:
    """관측 앞 두 칸. 원본은 이번 기간 수요를 아직 안 뽑아서 직전 기간 D, U 를 준다.

    D: 소매 수요. 그 기간에 고객이 원한 양. 행=기간, 열=판매 링크.
    U: 미충족. 그 기간에 못 준 양 = 수요 − 판매. D 와 같은 모양.
    Serial 은 판매 링크가 1→0 하나라 열은 항상 0.
    예: D[2, 0]=24, 재고가 충분하면 U[2, 0]=0. 재고 10이면 판매 10, U=14.
    """
    if period > 0:
        # 예: period=3 이면 어제 = 기간 2. D[2, 0], U[2, 0]
        return float(env.D[period - 1, 0]), float(env.U[period - 1, 0])
    return 0.0, 0.0  # reset 직후는 어제 수요가 없음


def observation_sentiment(env, period: int) -> float:
    """행동 직전에 관측되는 sentiment. 원본은 이전 step 종료 시점 값을 준다."""
    if period == 0:
        return 1.0  # 에피소드 시작 기본값
    return float(env.GW[period - 1])


def pipeline_rows_after_step(env, period: int) -> list[str]:
    """step 이 끝난 뒤 링크별 이동 중. 기간 시작 일정(pipeline_arrivals_at_start)과 짝.

    Y[period+1]: 오늘 발주를 더한 뒤의 이동 중 합. 오늘 도착분은 이미 빠짐.
    R[period]: 오늘 실제로 시킨 발주. 요청이 아니라 재고·용량으로 자른 양.
    arrives_at: 오늘 발주가 도착하는 기간. 파이프라인 전체가 끝나는 시점이 아님.
    """
    rows = []
    # lead_times 키는 발주 링크. (2,1) L=4, (3,2) L=4, (4,3) L=0
    for (src, dst), lead_time in env.network.lead_times.items():
        reorder_idx = env.network.reorder_map[(src, dst)]  # (2,1)=0. R, Y 의 열
        in_transit = float(env.Y[period + 1, reorder_idx])  # 예: 어제 20 + 오늘 20 − 오늘도착 0 = 40
        arrives_at = period + lead_time  # 예: period=0, L=4 → 기간 4. L=0 이면 오늘
        rows.append(
            f"  {src}->{dst}: 이번 발주 {env.R[period, reorder_idx]:.2f}, "
            f"다음 이동 중 Y={in_transit:.2f}, 이번 발주 도착 예정 기간={arrives_at}"
        )
    return rows  # 출력 문자열 3줄. 학습에는 안 씀


def collect_reward_terms(env, period: int) -> dict:
    """원본 CoreEnv._STEP과 같은 항목을 산식 문자열과 함께 모은다."""
    sr_lines, pc_lines, hc_lines, oc_lines, up_lines, fk_lines = [], [], [], [], [], []
    sales_revenue = purchase_cost = holding_cost = 0.0
    operating_cost = backlog_penalty = fixed_order_cost = 0.0

    for node_id in env.network.main_nodes:
        role = node_role(env, node_id)
        # SR: 아래로 보낸 양 × 그 링크 가격. 예: 소매 24 × 2.0 = 48
        for succ_idx, net_idx in env.network.succ_network_indices[node_id]:
            succ_id = env.network.idx_to_node[succ_idx]
            price = float(env.network.graph.edges[(node_id, succ_id)]["p"])
            shipped = float(env.S[period, net_idx])
            value = price * shipped
            sales_revenue += value
            sr_lines.append(
                f"  {role} {node_id}->{succ_id}: {price:.2f} × {shipped:.2f} = {value:.4f}"
            )

        # PC: 오늘 시킨 발주 × 사는 가격. 예: 유통이 소매에 20, p=1.5 → 30
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
            on_hand = float(env.X[period + 1, node_idx])  # step 끝난 뒤 창고
            on_hand_cost = hold_rate * on_hand  # 예: 소매 76 × 0.03
            holding_cost += on_hand_cost
            hc_lines.append(
                f"  {role} 보유 {node_id}: {hold_rate:.3f} × {on_hand:.2f} = {on_hand_cost:.4f}"
            )
            # 파이프라인 유지비 g×Y 는 보내는 쪽이 아니라 받는 쪽(node_id) HC 에 들어간다.
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
            op_rate = float(env.network.graph.nodes[node_id]["o"])  # 0.01
            yield_rate = float(env.network.graph.nodes[node_id]["v"])  # 1.0
            # OC = (o/v)×출하. o=가공비, v=수율. Serial 은 v=1 이라 0.01×공장출하.
            node_operating = (op_rate / yield_rate) * shipped
            operating_cost += node_operating
            oc_lines.append(
                f"  {role} {node_id}: ({op_rate:.3f} / {yield_rate:.3f}) × {shipped:.2f} = {node_operating:.4f}"
            )

        if node_id in env.network.retail:
            for succ_id in env.network.graph.successors(node_id):
                if (node_id, succ_id) in env.network.retail_map:
                    retail_idx = env.network.retail_map[(node_id, succ_id)]  # Serial 은 0
                    penalty_rate = float(env.network.graph.edges[(node_id, succ_id)]["b"])  # 0.1
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

    discount = float(env.alpha**period)  # Serial 기본 α=1 이라 항상 1
    # 세 노드 항목을 다 더한 뒤 한 번에 할인. 원본 step 보상과 같아야 함
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
    print("공식  한 노드 P = α^t × (SR − PC − OC − HC − UP − FK)")
    print("      r_t = P_소매 + P_유통 + P_공장   (여기 α=1 이라 할인은 없음)")
    print("  r_t=기간 보상, α=할인, SR=매출, PC=구매비, OC=운영비, HC=유지비, UP=품절벌금, FK=고정발주비(여기 0)")
    print("매출 SR (Sales Revenue) = Σ p × S")
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
    """한 기간 출력. period 는 step 직전 값(0~29). step 이 끝난 뒤 행렬 R,X,D 를 읽는다."""
    retail_idx = 0  # Serial 은 시장 링크가 하나. D, U 의 열 0
    retail_link = env.network.retail_links[retail_idx]  # (1, 0)
    retail_net_idx = env.network.network_map[retail_link]  # S 에서 소매 판매 칸
    opening = inventory_by_node(env, period)  # step 시작 창고 X[t]
    closing = inventory_by_node(env, period + 1)  # step 끝 창고 X[t+1]
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
    core = env.unwrapped  # Wrapper 를 벗겨 X, Y, R 행렬에 바로 접근

    print_environment(core)
    action = np.full(core.action_space.shape, args.order, dtype=np.float64)  # 예: [20, 20, 20]
    print("환경: GymInvMgmt/Serial-v0")
    print(f"이번 실행 정책: 매 기간 모든 링크에 {args.order} 발주")
    print(f"seed={args.seed}, 출력 기간={args.steps}")
    print()

    for _ in range(args.steps):
        # step 전에 읽어야 이 기간 번호가 맞다. step 끝나면 core.period 가 1 오른다.
        period = core.period
        _obs, reward, terminated, truncated, info = env.step(action)
        print_period(core, period, action, float(reward), info)
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
