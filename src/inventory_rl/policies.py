"""학습하지 않는 기준선 정책.

고정 발주: 매 기간 같은 수량. “아무 정보도 안 보고 평균만큼 넣는” 하한선.
재고 수준: 원본 examples/run_heuristic.py 의 NewsvendorPolicy를 Serial-v0에 맞게 옮긴 것.
           재고가 목표보다 모자라면 그 차이만 발주한다.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from inventory_rl.env import clip_order


class ConstantOrderPolicy:
    """세 발주 링크에 항상 같은 수량을 넣는다. 기본값은 수요 평균 20."""

    def __init__(self, order: float = 20.0) -> None:
        self.order = float(order)

    def get_action(self, obs: np.ndarray, env) -> np.ndarray:
        # obs 는 안 본다. 재고가 남아 있든 비었든 20을 넣는다.
        n_actions = env.action_space.shape[0]  # Serial-v0 는 3
        # 예: order=20 → [20, 20, 20]
        return clip_order(np.full(n_actions, self.order, dtype=np.float64))


class BaseStockPolicy:
    """노드마다 목표 재고를 정하고, 지금 가진 것과 목표의 차이만 발주한다.

    고정 발주가 매 기간 20 을 넣는 것과 달리, 여기 발주량은 매 기간 이렇게 계산한다.

        발주량 = max(0, 목표재고 S − 재고위치 IP)

    기호
        h   창고 유지비. 소매 0.03, 유통 0.02, 공장 0.01
        b   품절벌금. 소매만 0.1. 유통·공장은 마진(팔가격 − 살가격)으로 대신함
        L   그 노드로 오는 리드타임. 소매·유통 4, 공장 0
        X   지금 창고
        Y   지금 오는 중
        U   어제 못 준 양. 소매만
        d   어제 수요. 관측 첫 칸. 0 이면 20 을 씀 (reset 직후)

    1) 품절이 유지비보다 얼마나 아픈지 (critical ratio, 임계비율)
           cr = b / (b + h)
       0~1 사이. 품절이 더 아프면 1 에 가깝다.
       예: 유통 b=0.5, h=0.02 → cr = 0.5 / 0.52 ≈ 0.96

    2) 도착할 때까지 버틸 수요의 평균
           λ = d × (L + 1)
       예: d=20, L=4 → λ = 100

    3) 목표재고 S  (뉴스벤더. “수요가 Poisson(λ)일 때 cr 확률로 부족하지 않게”)
           S = Poisson(λ) 의 cr-분위수   = 코드의 poisson.ppf(cr, λ)
       cr 이 클수록 S 가 크다.

    4) 지금 재고위치 IP  (창고만 보면 오는 중을 빼고, 빚을 무시하게 됨)
           IP = X + Y − U
       예: X=80, Y=40, U=0 → IP=120

    5) 발주
           발주량 = max(0, S − IP)
       예: S=130, IP=120 → 10.  IP가 S 이상이면 0.

    d 는 환경의 진짜 평균 μ(t) 가 아니다. 기간 15 에 수요가 2 배가 되어도
    어제 숫자를 보기 전에는 S 를 미리 올려 두지 못한다.
    """

    def __init__(self, env, service_level: float = 0.95) -> None:
        # Gymnasium Wrapper 를 벗겨 원본 CoreEnv 행렬(X, Y, U)에 접근한다.
        core = env.unwrapped
        self.core = core
        network = core.network  # 공급망 지도. market=[0], rawmat=[4], retail=[1], factory=[3]
        # 시장(0)·원자재(4)는 발주 대상이 아니다. 소매·유통·공장만 남긴다.
        self.main_nodes = sorted(
            n for n in core.graph.nodes() if n not in network.market and n not in network.rawmat
        )
        self.node_info = {}
        for node in self.main_nodes:
            holding = core.graph.nodes[node].get("h", 0.1)  # 소매 h=0.03, 유통 0.02, 공장 0.01
            if node in network.retail:
                # 소매 링크의 미충족 벌점 b. 고객에게 못 주면 이만큼 깎인다.
                penalty = max(
                    # 소매→시장 (1,0)["b"]=0.1. 못 준 1단위당 벌점
                    (core.graph.edges[(node, k)].get("b", 0) for k in core.graph.successors(node)),
                    default=0.5,
                )
            else:
                # 위쪽 노드는 b 가 없다. 팔 가격 − 살 가격(마진)을 품절 비용 대신 쓴다.
                sell_prices = [
                    # 아래로 팔 때 가격. 유통이면 (2,1)["p"]=1.5
                    core.graph.edges.get((node, k), {}).get("p", 0) for k in core.graph.successors(node)
                ]
                buy_prices = [
                    # 위로 살 때 가격. 유통이면 (3,2)["p"]=1.0
                    core.graph.edges.get(edge, {}).get("p", 0)
                    for edge in network.reorder_links
                    if edge[1] == node
                ]
                # 예: 유통은 팔 1.5, 사 1.0 → penalty=0.5
                penalty = max(max(sell_prices, default=0) - min(buy_prices, default=0), 0.5)
            # cr 이 클수록 “품절이 유지비보다 아프다” → 목표 재고를 높게.
            # 예: penalty=0.5, holding=0.02 → cr=0.5/0.52 ≈ 0.96
            critical_ratio = penalty / (penalty + holding) if (penalty + holding) > 0 else service_level
            # 이 노드로 들어오는 발주 링크와, 그중 가장 긴 리드타임.
            incoming = [(i, edge) for i, edge in enumerate(network.reorder_links) if edge[1] == node]
            # 들어오는 링크의 리드타임. 소매·유통 (2,1)/(3,2)["L"]=4, 공장 (4,3)["L"]=0
            max_lead = max((core.graph.edges[edge]["L"] for _, edge in incoming), default=0)
            self.node_info[node] = {
                "cr": critical_ratio,
                "max_L": max_lead,  # 소매·유통은 L=4, 공장은 L=0
                "incoming": incoming,  # (행동 벡터의 몇 번째 칸, 링크)
            }

    def get_action(self, obs: np.ndarray, env) -> np.ndarray:
        core = env.unwrapped
        period = core.period
        obs_vec = np.asarray(obs, dtype=float).reshape(-1)
        last_demand = float(obs_vec[0]) if obs_vec.size else 0.0  # 관측 [0] = 직전 수요
        # reset 직후는 직전 수요가 0이다. 그때만 기본 평균 20을 쓴다.
        mean_demand = last_demand if last_demand > 0 else 20.0
        actions = np.zeros(len(core.network.reorder_links), dtype=np.float64)

        for node in self.main_nodes:
            info = self.node_info[node]
            node_idx = core.network.node_map[node]
            # 도착까지 L기간 + 오늘 1기간 동안 팔릴 양을 목표에 넣는다.
            # 예: 어제 수요 20, L=4 → 평균 20*5=100. 이 평균의 Poisson 에서 S 를 뽑음
            lead_demand = mean_demand * (info["max_L"] + 1)
            # ppf(cr, 평균): “이 수요를 cr 확률로 감당하려면 재고가 얼마 있어야 하나”
            target = poisson.ppf(info["cr"], lead_demand) if lead_demand > 0 else 0.0
            on_hand = core.X[period, node_idx]  # X=창고 재고. 지금 들고 있는 양
            in_transit = sum(core.Y[period, i] for i, _ in info["incoming"])  # Y=이동 중. 아직 안 도착한 합
            backlog = 0.0
            if node in core.network.retail and period > 0:
                # 소매만 고객 미충족 U 가 있다. 이미 빚진 양은 목표에서 빼야 한다.
                for k in core.graph.successors(node):
                    if (node, k) in core.network.retail_map:
                        backlog = core.U[period - 1, core.network.retail_map[(node, k)]]  # U=어제 미충족. 이미 빚진 양
            inventory_position = on_hand + in_transit - backlog  # 예: 80 + 40 - 0 = 120
            # 이미 S 이상이면 0. 모자라면 그 차이만 발주 (마이너스 발주는 없다).
            # 예: 목표 130, 위치 120 → 10 발주
            order_needed = max(0.0, float(target - inventory_position))
            n_suppliers = len(info["incoming"])
            if n_suppliers > 0:
                # Serial-v0 는 노드마다 공급자가 하나라 그냥 그 링크에 전부 넣는다.
                per_supplier = order_needed / n_suppliers
                for idx, _ in info["incoming"]:
                    actions[idx] = per_supplier
        return clip_order(actions)
