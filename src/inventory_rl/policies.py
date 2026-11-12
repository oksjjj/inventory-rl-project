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
        n_actions = env.action_space.shape[0]
        return clip_order(np.full(n_actions, self.order, dtype=np.float64))


class BaseStockPolicy:
    """재고 위치를 목표 S까지 채우는 정책.

    재고 위치 = 지금 들고 있는 것 + 아직 오는 중 − 이미 못 준 것
              = X + Y − U

    목표 S 는 뉴스벤더 식이다.
        임계비율 cr = 품절 비용 / (품절 비용 + 유지비)
        S = Poisson(리드타임 동안 수요).의 cr-분위수
    예: 유지비가 싸고 품절이 비싸면 cr≈1, S를 높게 잡는다.

    수요 평균은 시뮬레이터의 진짜 μ(t)가 아니라 관측의 직전 수요만 쓴다.
    그래서 기간 15에 수요가 갑자기 2배가 되어도, 그 사실을 미리 알 수는 없다.
    """

    def __init__(self, env, service_level: float = 0.95) -> None:
        # Gymnasium Wrapper 를 벗겨 원본 CoreEnv 행렬(X, Y, U)에 접근한다.
        core = env.unwrapped
        self.core = core
        network = core.network
        # 시장(0)·원자재(4)는 발주 대상이 아니다. 소매·유통·공장만 남긴다.
        self.main_nodes = sorted(
            n for n in core.graph.nodes() if n not in network.market and n not in network.rawmat
        )
        self.node_info = {}
        for node in self.main_nodes:
            holding = core.graph.nodes[node].get("h", 0.1)
            if node in network.retail:
                # 소매 링크의 미충족 벌점 b. 고객에게 못 주면 이만큼 깎인다.
                penalty = max(
                    (core.graph.edges[(node, k)].get("b", 0) for k in core.graph.successors(node)),
                    default=0.5,
                )
            else:
                # 위쪽 노드는 b 가 없다. 팔 가격 − 살 가격(마진)을 품절 비용 대신 쓴다.
                sell_prices = [
                    core.graph.edges.get((node, k), {}).get("p", 0) for k in core.graph.successors(node)
                ]
                buy_prices = [
                    core.graph.edges.get(edge, {}).get("p", 0)
                    for edge in network.reorder_links
                    if edge[1] == node
                ]
                penalty = max(max(sell_prices, default=0) - min(buy_prices, default=0), 0.5)
            # cr 이 클수록 “품절이 유지비보다 아프다” → 목표 재고를 높게.
            critical_ratio = penalty / (penalty + holding) if (penalty + holding) > 0 else service_level
            # 이 노드로 들어오는 발주 링크와, 그중 가장 긴 리드타임.
            incoming = [(i, edge) for i, edge in enumerate(network.reorder_links) if edge[1] == node]
            max_lead = max((core.graph.edges[edge]["L"] for _, edge in incoming), default=0)
            self.node_info[node] = {
                "cr": critical_ratio,
                "max_L": max_lead,
                "incoming": incoming,  # (행동 벡터의 몇 번째 칸, 링크)
            }

    def get_action(self, obs: np.ndarray, env) -> np.ndarray:
        core = env.unwrapped
        period = core.period
        obs_vec = np.asarray(obs, dtype=float).reshape(-1)
        last_demand = float(obs_vec[0]) if obs_vec.size else 0.0
        # reset 직후는 직전 수요가 0이다. 그때만 기본 평균 20을 쓴다.
        mean_demand = last_demand if last_demand > 0 else 20.0
        actions = np.zeros(len(core.network.reorder_links), dtype=np.float64)

        for node in self.main_nodes:
            info = self.node_info[node]
            node_idx = core.network.node_map[node]
            # 주문이 도착하는 동안(L) 플러스 이번 기간(1)만큼의 수요를 커버한다.
            lead_demand = mean_demand * (info["max_L"] + 1)
            # Poisson 누적분포가 cr 이 되는 지점 = 목표 재고 S.
            target = poisson.ppf(info["cr"], lead_demand) if lead_demand > 0 else 0.0
            on_hand = core.X[period, node_idx]  # 지금 창고에 있는 양
            in_transit = sum(core.Y[period, i] for i, _ in info["incoming"])  # 오는 중
            backlog = 0.0
            if node in core.network.retail and period > 0:
                # 소매만 고객 미충족 U 가 있다. 이미 빚진 양은 목표에서 빼야 한다.
                for k in core.graph.successors(node):
                    if (node, k) in core.network.retail_map:
                        backlog = core.U[period - 1, core.network.retail_map[(node, k)]]
            inventory_position = on_hand + in_transit - backlog
            # 이미 S 이상이면 0. 모자라면 그 차이만 발주 (마이너스 발주는 없다).
            order_needed = max(0.0, float(target - inventory_position))
            n_suppliers = len(info["incoming"])
            if n_suppliers > 0:
                # Serial-v0 는 노드마다 공급자가 하나라 그냥 그 링크에 전부 넣는다.
                per_supplier = order_needed / n_suppliers
                for idx, _ in info["incoming"]:
                    actions[idx] = per_supplier
        return clip_order(actions)
