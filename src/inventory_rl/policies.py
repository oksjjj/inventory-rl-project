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
       오늘 시킨 발주는 L기간 뒤에 도착한다. 그 사이 수요는 L번이고,
       도착한 그 기간에도 수요가 한 번 더 있다. 그래서 +1.
       예: 기간 0 에 시키면 기간 4 에 도착 (L=4).
           수요가 나는 날 = 기간 0,1,2,3,4 → 5번 = L+1.
           d=20 이면 λ = 20×5 = 100.
       L=0 공장: 같은 기간에 도착 → λ = d×1. 오늘 수요만 보면 됨.

    3) 목표재고 S
       왜 Poisson 인가.
       수요는 0,1,2,... 개수다. 한 기간 손님 수를 세는 기본 모형이 Poisson.
       Poisson(20) 은 숫자 20 이 아니라 분포다. 기댓값만 20 으로 고정.
       식 P(수요=k)=e^{-20} 20^k / k! 은 확률질량(PMF)이다. 밀도(PDF)가 아님.
       그 확률대로 정수를 뽑는다. 20 근처가 잘 나오고 음수는 없다.
       환경은 poisson.rvs 로 수요를 샘플링한다. seed 가 같으면 같은 열이 나온다.
       yaml demand_dist=poisson. trend+shock 은 평균 μ 만 바꾸고 분포는 Poisson.

       한 기간 ~ Poisson(d) 이고 독립이면 (L+1)기간 합 ~ Poisson(λ), λ=d×(L+1).

       뉴스벤더 식 (이 정책이 실제로 푸는 것):
           P(총수요 ≤ S) ≥ cr     →     S = poisson.ppf(cr, λ)
       ppf = Percent Point Function (분위수 함수). CDF(누적분포함수) 의 역함수.
       총수요 ~ Poisson(λ). 재고 S 로 그 수요를 cr 확률 이상 감당하게.

       poisson.ppf 는 수요를 뽑지 않는다. 위 식을 만족하는 가장 작은 S 를 계산한다.
       왼쪽부터 P(총수요≤0), P(≤1), P(≤2), ... 를 더해 가다가 합이 cr 이상이 되는 첫 S.
       λ, cr 이 같으면 항상 같은 S (랜덤 아님).
       작은 예: Poisson(2), cr=0.95
           P(≤4)≈0.947 < 0.95,  P(≤5)≈0.983 ≥ 0.95  →  S=5
       여기선 λ=100, cr≈0.96 이라 S 가 100 보다 조금 크다. cr 이 1 에 가까울수록 S 가 커진다.

    4) 지금 재고위치 IP  (창고만 보면 오는 중을 빼고, 빚을 무시하게 됨)
           IP = X + Y − U
       예: X=80, Y=40, U=0 → IP=120

    5) 발주
           발주량 = max(0, S − IP)
       예: S=130, IP=120 → 10.  IP가 S 이상이면 0.

    이 정책은 시뮬레이터 안의 진짜 평균 μ(t) 를 모른다. d 는 어제 나온 수요 숫자뿐.
    학습 기본 수요 trend+shock 는 기간 15 부터 평균이 약 2 배가 된다.
    기간 15 를 아직 안 겪었으면 어제는 여전히 ~20 이라 S 도 낮게 잡는다.
    큰 수요가 관측된 다음 기간에야 d 가 커지고 S 를 올린다. 미리 알 수 없다.
    """

    def __init__(self, env, service_level: float = 0.95) -> None:
        # Gymnasium Wrapper 를 벗겨 원본 CoreEnv 행렬(X, Y, U)에 접근한다.
        core = env.unwrapped
        self.core = core
        # core.network: 누가 누구와 연결됐는지 (재고 수량 X 가 아님).
        # 역할 속성은 “그 역할인 노드 번호” list. Serial-v0 실제 값:
        #   network.market  = [0]     시장 (수요가 나가는 곳)
        #   network.retail  = [1]     소매
        #   network.distrib = [1, 2]  vendor 분류: I0 있고 C 없음. 우리가 말하는 유통은 2 뿐.
        #                            소매 1 도 이 리스트에 들어감 (retail 과 겹침)
        #   network.factory = [3]     공장
        #   network.rawmat  = [4]     원자재
        # 예: if node_id in network.market  →  0 인지 본다. 재고가 0 인 것과 무관.
        network = core.network
        # 시장(0)·원자재(4)는 발주 대상이 아니다. 소매·유통·공장만 남긴다.
        self.main_nodes = sorted(
            n for n in core.graph.nodes() if n not in network.market and n not in network.rawmat
        )
        self.node_info = {}
        for node in self.main_nodes:
            holding = core.graph.nodes[node].get("h", 0.1)  # 소매 h=0.03, 유통 0.02, 공장 0.01
            if node in network.retail:
                # 소매의 품절비용 penalty = 고객 링크의 벌점 b 중 최댓값.
                # successors(1) = [0]  (소매가 파는 곳 = 시장)
                # edges[(1, 0)]["b"] = 0.1
                # max(0.1) = 0.1. 링크가 하나도 없으면 default 0.5
                penalty = max(
                    (core.graph.edges[(node, k)].get("b", 0) for k in core.graph.successors(node)),
                    default=0.5,
                )
            else:
                # 위쪽 노드는 b 가 없다. 팔 가격 − 살 가격(마진)을 품절 비용 대신 쓴다.
                sell_prices = [
                    # 아래로 팔 때 가격. 유통이면 (2,1)["p"]=1.5
                    core.graph.edges.get((node, k), {}).get("p", 0) for k in core.graph.successors(node)
                ]
                # 이 노드가 윗단계에서 사 올 때 내는 가격 p 들.
                # reorder_links 의 edge=(보내는 곳, 받는 곳). edge[1]==node 면 “나한테 오는 발주 링크”.
                # 유통 node=2: 조건에 맞는 링크는 (3,2)뿐 → p=1.0 → buy_prices=[1.0]
                # 공장 node=3: (4,3) → p=0.5 → buy_prices=[0.5]
                buy_prices = [
                    core.graph.edges.get(edge, {}).get("p", 0)
                    for edge in network.reorder_links
                    if edge[1] == node
                ]
                # 환경 보상에는 유통·공장 U, b 가 없다. 소매만 UP=b×U.
                # 그래도 이 정책은 모든 노드에 cr = penalty/(penalty+h) 가 필요해서
                # “못 내려보내면 잃는 마진”(팔가격 − 살가격)을 품절비용 대신 쓴다.
                # 예: 유통 팔 1.5 − 사 1.0 = 0.5. 마진이 0 이하면 바닥 0.5
                penalty = max(max(sell_prices, default=0) - min(buy_prices, default=0), 0.5)
            # cr 이 클수록 “품절이 유지비보다 아프다” → 목표 재고를 높게.
            # 예: penalty=0.5, holding=0.02 → cr=0.5/0.52 ≈ 0.96
            critical_ratio = penalty / (penalty + holding) if (penalty + holding) > 0 else service_level
            # 이 노드로 들어오는 발주 링크만. (action 칸 번호 i, 링크 튜플)
            # edge[1]==node 가 “도착지가 나”. 소매1 → [(0, (2,1))], 유통2 → [(1, (3,2))]
            incoming = [(i, edge) for i, edge in enumerate(network.reorder_links) if edge[1] == node]
            # 위에서 모은 링크의 L 중 최댓값. Serial 은 링크가 하나라 그 L 그대로.
            # 소매·유통 L=4, 공장 L=0. 링크가 없으면 default 0
            max_lead = max((core.graph.edges[edge]["L"] for _, edge in incoming), default=0)
            # 노드마다 매 기간 안 바뀌는 것만 미리 저장. get_action 이 매 기간 꺼내 쓴다.
            # 목표 수요 λ 자체는 여기 없다. λ = (어제 수요 d)×(max_L+1) 이라 d 는 관측에서 온다.
            # cr: 뉴스벤더 분위수. S = poisson.ppf(cr, λ). 유통 ≈0.96
            # max_L: λ 의 (L+1). 소매·유통 4, 공장 0
            # incoming: Y 합산 + 발주를 행동 벡터 몇 번째 칸에 넣을지. 소매 [(0,(2,1))]
            self.node_info[node] = {
                "cr": critical_ratio,
                "max_L": max_lead,
                "incoming": incoming,
            }

    def get_action(self, obs: np.ndarray, env) -> np.ndarray:
        # Wrapper 를 벗겨 CoreEnv. X, Y, U, period 를 직접 읽으려고.
        core = env.unwrapped
        # 지금이 몇 번째 기간인지. reset 직후 0. step 안에서 1 증가.
        # 아래 X[period], Y[period] 의 행 번호로 쓴다.
        period = core.period
        # obs 가 (1,15) 이어도 (15,) 로 펼친다. 신경망과 달리 여기선 정규화 안 함.
        obs_vec = np.asarray(obs, dtype=float).reshape(-1)
        # 원본 관측 첫 칸 = 어제 수요 D. 빈 배열이면 0.
        last_demand = float(obs_vec[0]) if obs_vec.size else 0.0
        # reset 직후는 어제 수요가 아직 없어서 0. 그때만 기본 평균 20.
        # 기간 1부터는 어제 나온 숫자. 예: 18 → 18,  0 → 20
        mean_demand = last_demand if last_demand > 0 else 20.0
        # 발주 3칸을 0으로 시작. 아래 루프가 소매·유통·공장 칸을 채운다.
        # reorder_links = [(2,1), (3,2), (4,3)] → 길이 3 → [0, 0, 0]
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
            backlog = 0.0  # 유통·공장·기간 0 은 빚이 없음. 아래는 소매만
            if node in core.network.retail and period > 0:
                # 소매만 고객 미충족 U 가 있다. 이미 빚진 양은 IP 에서 뺀다.
                # period>0: U[period-1] 을 읽으므로 reset 직후(period=0)는 어제가 없다.
                for k in core.graph.successors(node):
                    if (node, k) in core.network.retail_map:
                        # U 행 = 어제, 열 = 판매 링크.
                        # 지금 period 의 step 은 아직 안 돌아서 U[period] 는 비어 있다.
                        # 그래서 어제 못 준 양 U[period-1] 을 읽는다.
                        # retail_map: 판매 링크 → D,U 열. Serial 은 (1,0)→0 하나뿐.
                        # 예: node=1, k=0, period=3, 어제 14 못 줌
                        #     → U[2, retail_map[(1,0)]] = U[2, 0] = 14
                        # IP = X+Y−14 가 되어 목표까지 14 더 발주하게 된다.
                        backlog = core.U[period - 1, core.network.retail_map[(node, k)]]
            inventory_position = on_hand + in_transit - backlog  # 예: 80 + 40 - 0 = 120
            # 이미 S 이상이면 0. 모자라면 그 차이만 발주 (마이너스 발주는 없다).
            # 예: 목표 130, 위치 120 → 10 발주
            order_needed = max(0.0, float(target - inventory_position))
            n_suppliers = len(info["incoming"])
            if n_suppliers > 0:
                # order_needed 는 노드 창고 기준 총량 (S − IP). 링크마다 그 양을 넣으면
                # 합이 n배가 되어 S 를 넘는다. 그래서 합이 order_needed 가 되게 나눈다.
                # 예: 모자란 30, 공급자 2명 → 각 15. 각 30 이면 총 60.
                # Serial 은 공급자 1명 → 나누기 1, 그 링크에 전부.
                # 균등 분할은 원본 휴리스틱의 단순 규칙이다. L·가격이 달라도 구분하지 않는다.
                per_supplier = order_needed / n_suppliers
                # incoming = [(행동 벡터 칸 번호, 링크)]. __init__ 에서 reorder_links 순번 i.
                # Serial reorder_links=[(2,1),(3,2),(4,3)] 이라
                #   소매1 → [(0, (2,1))]  idx=0  유통→소매
                #   유통2 → [(1, (3,2))]  idx=1  공장→유통
                #   공장3 → [(2, (4,3))]  idx=2  원자재→공장
                # _ 는 링크 튜플. 칸 번호만 있으면 되어 안 쓴다.
                # 예: 소매 order_needed=10 → actions[0]=10 → [10, ?, ?]
                for idx, _ in info["incoming"]:
                    actions[idx] = per_supplier
        return clip_order(actions)
